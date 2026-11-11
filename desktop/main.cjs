const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { writeFile } = require('node:fs/promises');
const { WorkerClient, validateRequest } = require('./worker-client.cjs');

const root = path.resolve(__dirname, '..');
const page = pathToFileURL(path.join(__dirname, 'index.html')).href;
const worker = new WorkerClient(root);
let mainWindow;
if (process.env.MIRRORLAB_USER_DATA) app.setPath('userData', process.env.MIRRORLAB_USER_DATA);

function trusted(event) {
  if (!mainWindow || event.sender !== mainWindow.webContents || event.senderFrame?.url !== page) {
    throw new Error('Недопустимый источник команды.');
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1480, height: 980, minWidth: 1000, minHeight: 700,
    show: process.env.MIRRORLAB_HEADLESS !== '1',
    backgroundColor: '#ffffff', title: 'MirrorLab · Конфигурация зеркал',
    autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  mainWindow.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  mainWindow.webContents.on('will-navigate', (event) => event.preventDefault());
  mainWindow.webContents.session.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  mainWindow.on('closed', () => { mainWindow = null; worker.stop(); });
  mainWindow.loadFile(path.join(__dirname, 'index.html'));
}

app.whenReady().then(() => {
  ipcMain.handle('mirrorlab:request', (event, method, payload = {}) => {
    trusted(event);
    const clean = validateRequest(method, payload);
    if (method === 'cancel') { worker.stop(); return { cancelled: true }; }
    return worker.request(method, clean);
  });
  ipcMain.handle('mirrorlab:save', async (event, data) => {
    trusted(event);
    const text = JSON.stringify(data, null, 2);
    if (typeof text !== 'string' || text.length > 100_000_000) throw new Error('Слишком большой файл экспорта.');
    const { canceled, filePath } = await dialog.showSaveDialog(mainWindow, {
      title: 'Сохранить результат расчёта', defaultPath: 'mirrorlab-result.json',
      filters: [{ name: 'Результат MirrorLab', extensions: ['json'] }],
    });
    if (canceled || !filePath) return { saved: false };
    await writeFile(filePath, text, 'utf8');
    return { saved: true, path: filePath };
  });
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});
app.on('before-quit', () => worker.stop());
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
