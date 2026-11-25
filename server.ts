import 'dotenv/config';
import express from 'express';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';
import { createServer as createViteServer } from 'vite';
import { createZipBuffer, type ZipEntryInput } from './server/zipBuilder.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const isProduction = process.env.NODE_ENV === 'production';

function getAiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return null;
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '1mb' }));

  app.post('/api/comfort', async (req, res) => {
    try {
      const body = req.body as {
        content?: unknown;
        title?: unknown;
        moodLabel?: unknown;
        feelingLabel?: unknown;
      };

      const rawTitle = typeof body.title === 'string' ? body.title.trim() : '';
      const rawContent = typeof body.content === 'string' ? body.content.trim() : '';
      const moodLabel = typeof body.moodLabel === 'string' ? body.moodLabel.trim().slice(0, 40) : '';
      const feelingLabel = typeof body.feelingLabel === 'string' ? body.feelingLabel.trim().slice(0, 60) : '';

      if (!rawTitle && !rawContent) {
        res.status(400).json({ error: '일기 내용을 한 줄 이상 입력해 주세요.' });
        return;
      }

      if (rawTitle.length > 300 || rawContent.length > 5000) {
        res.status(400).json({ error: '입력 내용이 너무 깁니다.' });
        return;
      }

      const ai = getAiClient();
      if (!ai) {
        res.status(500).json({
          error: 'GEMINI_API_KEY가 설정되지 않았습니다. Settings > Secrets 패널을 확인해 주세요.',
        });
        return;
      }

      const contextLines: string[] = [];
      if (moodLabel) contextLines.push(`오늘의 날씨: ${moodLabel}`);
      if (feelingLabel) contextLines.push(`오늘의 기분: ${feelingLabel}`);
      if (rawTitle) contextLines.push(`일기 제목: ${rawTitle}`);
      if (rawContent) contextLines.push(`일기 내용:\n${rawContent}`);

      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: contextLines.join('\n'),
        config: {
          systemInstruction:
            '당신은 다정하고 사려 깊은 감성 일기장의 동반자입니다. 사용자가 오늘 기록한 일기, 날씨, 기분을 읽고 진심이 담긴 따뜻한 위로와 공감의 한 마디를 한국어로 2~3문장(약 80~140자) 이내로 건네주세요. 과장되거나 가르치려는 어조는 피하고, 조용히 마음을 토닥여주는 부드러운 해요체로 작성해 주세요. 따옴표나 불필요한 서두 없이 위로의 말만 자연스럽게 답해 주세요.',
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        },
      });

      const comfortText = response.text?.trim();
      if (!comfortText) {
        res.status(502).json({ error: '위로 메시지를 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.' });
        return;
      }

      res.json({ message: comfortText });
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : String(err);
      console.error('Gemini comfort generation error:', errMsg);

      if (errMsg.includes('403') || errMsg.includes('PERMISSION_DENIED') || errMsg.includes('API_KEY_INVALID')) {
        res.status(403).json({
          error: 'API 키 인증에 실패했습니다. Settings > Secrets 패널에서 GEMINI_API_KEY를 확인해 주세요.',
        });
        return;
      }
      if (errMsg.includes('429') || errMsg.includes('RESOURCE_EXHAUSTED')) {
        res.status(429).json({
          error: '요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.',
        });
        return;
      }

      res.status(500).json({ error: 'AI 위로 한 마디를 불러오는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.' });
    }
  });

  const ROOT_SOURCE_ALLOWLIST = [
    'index.html',
    'package.json',
    'tsconfig.json',
    'vite.config.ts',
    'metadata.json',
    '.env.example',
    '.gitignore',
    'server.ts',
  ];
  const DIR_SOURCE_ALLOWLIST = ['src', 'server'];

  async function collectSafeSourceFiles(rootDir: string): Promise<ZipEntryInput[]> {
    const collected: ZipEntryInput[] = [];
    const baseWithSep = rootDir.endsWith(path.sep) ? rootDir : `${rootDir}${path.sep}`;

    for (const fileName of ROOT_SOURCE_ALLOWLIST) {
      const fullPath = path.resolve(rootDir, fileName);
      if (!fullPath.startsWith(baseWithSep)) continue;
      try {
        const stat = await fs.lstat(fullPath);
        if (stat.isFile() && !stat.isSymbolicLink()) {
          const data = await fs.readFile(fullPath);
          collected.push({ name: `moodlog/${fileName}`, data });
        }
      } catch {
        // skip optional root files if absent
      }
    }

    async function walkDir(dirPath: string, relPrefix: string) {
      const entries = await fs.readdir(dirPath, { withFileTypes: true });
      for (const entry of entries) {
        if (entry.isSymbolicLink()) continue;
        if (entry.name.startsWith('.')) continue;
        const fullPath = path.resolve(dirPath, entry.name);
        if (!fullPath.startsWith(baseWithSep)) continue;
        const relPath = `${relPrefix}/${entry.name}`;
        if (entry.isDirectory()) {
          await walkDir(fullPath, relPath);
        } else if (entry.isFile()) {
          const data = await fs.readFile(fullPath);
          collected.push({ name: `moodlog/${relPath}`, data });
        }
      }
    }

    for (const dirName of DIR_SOURCE_ALLOWLIST) {
      const dirPath = path.resolve(rootDir, dirName);
      if (!dirPath.startsWith(baseWithSep)) continue;
      try {
        const stat = await fs.lstat(dirPath);
        if (stat.isDirectory() && !stat.isSymbolicLink()) {
          await walkDir(dirPath, dirName);
        }
      } catch {
        // skip if absent
      }
    }

    return collected;
  }

  app.get('/api/download-code', async (_req, res) => {
    try {
      const files = await collectSafeSourceFiles(__dirname);
      if (files.length === 0) {
        res.status(404).json({ error: '다운로드할 소스 파일을 찾지 못했습니다.' });
        return;
      }
      const zipBuf = createZipBuffer(files);
      res.setHeader('Content-Type', 'application/zip');
      res.setHeader('Content-Disposition', 'attachment; filename="MoodLog-source.zip"');
      res.setHeader('Content-Length', String(zipBuf.length));
      res.send(zipBuf);
    } catch (err: unknown) {
      console.error('Source code zip error:', err instanceof Error ? err.message : String(err));
      res.status(500).json({ error: '소스 코드 압축 중 문제가 발생했습니다.' });
    }
  });

  if (isProduction) {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*splat', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`MoodLog server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
