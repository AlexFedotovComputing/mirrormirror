const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('mirrorlab', {
  request: (method, payload = {}) => ipcRenderer.invoke('mirrorlab:request', method, payload),
  saveResult: (data) => ipcRenderer.invoke('mirrorlab:save', data),
});
