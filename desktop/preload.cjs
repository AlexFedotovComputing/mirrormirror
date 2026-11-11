const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('mirrorlab', {
  request: (method, payload = {}) => ipcRenderer.invoke('mirrorlab:request', method, payload),
  saveResult: (data) => ipcRenderer.invoke('mirrorlab:save', data),
  onProgress: (callback) => {
    const listener = (_event, message) => callback(message);
    ipcRenderer.on('mirrorlab:progress', listener);
    return () => ipcRenderer.removeListener('mirrorlab:progress', listener);
  },
});
