import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const distPath = join(__dirname, 'frontend/dist');

// Serve static files only if they exist
app.use((req, res, next) => {
  const filePath = join(distPath, req.path);
  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    express.static(distPath)(req, res, next);
  } else {
    next();
  }
});

// Serve API (if backend is available)
app.use('/api', (req, res) => {
  res.status(503).json({ error: 'API not configured in this deployment' });
});

// SPA fallback - serve index.html for all non-API routes
app.get('*', (req, res) => {
  res.sendFile(join(distPath, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
