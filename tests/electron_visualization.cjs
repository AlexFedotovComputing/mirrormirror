/* End-to-end current tracer -> visualization tab, with real ideal/manual data. */
const path = require('node:path');
const fs = require('node:fs/promises');
const assert = require('node:assert/strict');
const { _electron } = require('playwright-core');

(async () => {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, 'artifacts', 'ui', 'visualization');
  await fs.mkdir(output, { recursive: true });
  const env = { ...process.env, MIRRORLAB_HEADLESS:'1', MIRRORLAB_USER_DATA:path.join(output,'profile') };
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({ executablePath:path.join(root,'node_modules/electron/dist/electron.exe'), args:[root],cwd:root,env });
  try {
    const page=await app.firstWindow(); page.setDefaultTimeout(120000);
    const errors=[]; page.on('pageerror', e=>errors.push(e.message));
    await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].webContents.setBackgroundThrottling(false));
    await page.waitForFunction(()=>!document.querySelector('#controls').disabled);
    await page.click('.nav-button[data-tab="distributions"]');
    assert.equal(await page.locator('#viz-empty').isVisible(),true);
    await page.click('.nav-button[data-tab="static"]');
    await page.fill('#ray-count','217');
    await page.fill('#cpu-workers','2');
    const trace=async kind=>{
      await page.selectOption('#kind',kind); await page.click('#run');
      await page.waitForFunction(()=>!document.querySelector('#controls').disabled && /Конфигурация построена|Расчёт не завершён/.test(document.querySelector('#run-status').textContent));
      assert.equal(await page.locator('#run-status').innerText(),'Конфигурация построена',await page.locator('#notice').innerText());
    };
    const ready=()=>page.waitForFunction(()=>{
      const d=document.querySelector('#viz-content').dataset;
      return d.rendered && d.rendered===d.pending;
    });
    const select=async(id,value)=>{
      await page.locator('#viz-content').evaluate(e=>delete e.dataset.rendered);
      await page.selectOption(id,value); await ready();
    };
    const view=async name=>{
      await page.locator('#viz-content').evaluate(e=>delete e.dataset.rendered);
      await page.click(`[data-viz-view="${name}"]`); await ready();
    };
    const capture=async name=>{
      await ready();
      const target = name === 'cameras' ? '#viz-cameras' : name === 'scene' ? '#viz-scene' : `#viz-${name}-view`;
      await page.locator(target).evaluate(e=>e.scrollIntoView({block:'start'}));
      const png=await app.evaluate(async({BrowserWindow})=>(await BrowserWindow.getAllWindows()[0].webContents.capturePage(undefined,{stayHidden:true})).toPNG().toString('base64'));
      await fs.writeFile(path.join(output,`${name}.png`),Buffer.from(png,'base64'));
    };
    await trace('ideal');
    await page.click('.nav-button[data-tab="distributions"]'); await ready();
    await select('#viz-side','WEST'); await select('#viz-rod','2'); await select('#viz-bundle','3');
    const ids=await page.locator('#viz-scene').evaluate(e=>e.data.filter(t=>t.meta?.mirror_id).map(t=>t.meta.mirror_id));
    assert.ok(ids.length); assert.ok(ids.every(id=>id.startsWith('2_3_')));
    await capture('scene');
    await select('#viz-side','both');
    await view('endcaps');
    assert.equal(await page.locator('#viz-roc-hover').isChecked(),true);
    assert.equal(await page.locator('#viz-roc-landmarks').isChecked(),true);
    await page.waitForFunction(()=>typeof session !== 'undefined' &&
      session.trace?.visualization?.sides.every(side=>side.endcap?.basis_survey?.length===2) &&
      [...document.querySelectorAll('#viz-endcaps .js-plotly-plot')].every(plot=>
        plot.data.some(t=>t.meta?.mirror_id && t.customdata?.some(row=>row?.length===11))));
    assert.equal(await page.locator('#viz-endcaps .js-plotly-plot').count(),2);
    for (const p of await page.locator('#viz-endcaps .js-plotly-plot').all()) {
      const data=await p.evaluate(e=>e.data);
      assert.equal(data.filter(t=>t.meta?.layer==='roc').length,12);
      assert.ok(data.filter(t=>t.meta?.mirror_id).every(t=>t.meta.mirror_id.startsWith('2_3_')));
    }
    await select('#viz-side','WEST');
    assert.equal(await page.locator('#viz-endcaps .js-plotly-plot').count(),1);
    await page.uncheck('#viz-roc-outlines'); await ready();
    assert.equal(await page.locator('#viz-endcaps .js-plotly-plot').evaluate(e=>e.data.filter(t=>t.meta?.layer==='roc').length),0);
    await page.check('#viz-roc-outlines'); await ready();
    const hoverProjectedRay = async () => {
      const target = await page.locator('#viz-endcaps .js-plotly-plot').first().evaluate(plot=>{
        const xa=plot._fullLayout.xaxis, ya=plot._fullLayout.yaxis;
        const rect=plot.getBoundingClientRect();
        const scaleX=rect.width/plot._fullLayout.width, scaleY=rect.height/plot._fullLayout.height;
        for (const trace of plot.data) {
          if (!trace.meta?.mirror_id || trace.visible===false || trace.visible==='legendonly') continue;
          for (let i=0;i<trace.x.length-1;i+=3) {
            if (!trace.customdata?.[i] || !trace.customdata?.[i+1]) continue;
            const x=xa._offset+xa.l2p((trace.x[i]+trace.x[i+1])/2);
            const y=ya._offset+ya.l2p((trace.y[i]+trace.y[i+1])/2);
            if (x<=xa._offset+8 || x>=xa._offset+xa._length-8 ||
                y<=ya._offset+8 || y>=ya._offset+ya._length-8) continue;
            plot.dispatchEvent(new PointerEvent('pointermove',{bubbles:true,buttons:0,
              clientX:rect.left+x*scaleX,clientY:rect.top+y*scaleY}));
            return trace.meta.mirror_id;
          }
        }
        return null;
      });
      assert.ok(target,'A projected ray midpoint must lie inside the visible axes.');
      await page.waitForFunction(()=>[...document.querySelectorAll('.mirror-ray-tooltip')].some(t=>!t.hidden));
      const text=await page.locator('.mirror-ray-tooltip:not([hidden])').innerText();
      assert.match(text,/Луч 2_3_[1-7]/);
      assert.match(text,/бандл 3/);
      assert.match(text,/ID трассировки: \d+/);
      assert.match(text,/X = .+ м\nY = .+ м\nZ = .+ м/);
    };
    await hoverProjectedRay();
    await page.uncheck('#viz-roc-hover'); await ready();
    assert.equal(await page.locator('.mirror-ray-tooltip').count(),0);
    await page.check('#viz-roc-hover'); await ready();
    await hoverProjectedRay();
    await select('#viz-side','both');
    await capture('endcaps');
    const endcapPng = await page.locator('#viz-endcaps .js-plotly-plot').first().evaluate(e=>Plotly.toImage(e,{format:'png',width:850,height:850,scale:1}));
    await fs.writeFile(path.join(output,'endcap-plot.png'),Buffer.from(endcapPng.split(',')[1],'base64'));
    await view('cameras');
    assert.equal(await page.locator('#viz-cameras .js-plotly-plot').count(),8);
    assert.equal(await page.locator('#viz-camera-table tbody tr').count(),8);
    await page.click('#viz-reference');
    await page.waitForFunction(()=>document.querySelector('#viz-reference-note').textContent.includes('опорный идеальный'));
    await select('#viz-side','WEST');
    await capture('cameras');
    await view('gas');
    assert.equal(await page.locator('#viz-gas-table tbody tr').count(),7);
    assert.equal(await page.locator('#viz-unwraps .js-plotly-plot').count(),1);
    await select('#viz-mirror','4');
    assert.equal(await page.locator('#viz-gas-table tbody tr').count(),1);
    await select('#viz-mirror','all');
    await capture('gas');
    await select('#viz-rod','all'); await select('#viz-bundle','all'); await select('#viz-side','both');
    assert.equal(await page.locator('#viz-gas-table tbody tr').count(),224);
    assert.equal(await page.locator('#viz-unwraps .js-plotly-plot').count(),32);
    await select('#viz-side','WEST');
    await view('source');
    assert.ok(await page.locator('#viz-angles tbody tr').count()>4);
    assert.equal(await page.locator('#viz-source .js-plotly-plot').count(),1);
    await capture('source');
    await page.uncheck('#viz-source-map'); await ready();
    assert.equal(await page.locator('#viz-source .js-plotly-plot').evaluate(e=>e.data.some(t=>t.type==='heatmap')),false);
    const sourcePng = await page.locator('#viz-source .js-plotly-plot').evaluate(e=>Plotly.toImage(e,{format:'png',width:850,height:650,scale:1}));
    await fs.writeFile(path.join(output,'source-plot.png'),Buffer.from(sourcePng.split(',')[1],'base64'));
    await page.click('.nav-button[data-tab="static"]');
    await trace('real');
    await page.click('.nav-button[data-tab="distributions"]'); await view('cameras');
    assert.equal(await page.locator('#viz-reference').isDisabled(),true);
    assert.match(await page.locator('#viz-context').innerText(),/Реальная/);
    assert.match(await page.locator('#viz-reference-note').innerText(),/опорный идеальный/);
    await select('#viz-side','EAST');
    assert.equal(await page.locator('#viz-camera-table tbody tr').count(),4);
    await page.click('.nav-button[data-tab="static"]'); await page.fill('#ray-count','1000');
    await page.click('.nav-button[data-tab="distributions"]');
    assert.equal(await page.locator('#viz-stale').isVisible(),true);
    assert.deepEqual(errors,[]);
    console.log('Visualization: ideal/manual, both sides, camera maps, reference centres, filters, unwraps and source passed.');
  } finally { await app.close(); }
})().catch(error=>{ console.error(error);process.exitCode=1; });
