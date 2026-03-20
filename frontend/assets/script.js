const EC = {
  PRODUCT_NAME: {bg:'rgba(37,99,235,.1)',color:'#1e40af',border:'rgba(37,99,235,.3)'},
  PRODUCT_TYPE: {bg:'rgba(124,58,237,.1)',color:'#6d28d9',border:'rgba(124,58,237,.3)'},
  PRICE:        {bg:'rgba(5,150,105,.1)',color:'#047857',border:'rgba(5,150,105,.3)'},
  LOCATION:     {bg:'rgba(190,24,93,.1)',color:'#be123c',border:'rgba(190,24,93,.3)'}
};

function esc(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

window.run = async function(){
  const txt = document.getElementById('txt');
  const s = txt ? txt.value.trim() : '';
  if(!s){ if(txt)txt.focus(); return; }
  
  const btn = document.getElementById('btn-run');
  if(!btn)return;
  btn.disabled = true;
  btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="animation:spin .7s linear infinite"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg><span>Processing...</span>';
  
  const prog = document.getElementById('prog');
  if(prog)prog.classList.add('on');
  
  const errDiv = document.getElementById('err');
  if(errDiv)errDiv.innerHTML='';
  
  // Hide panels
  document.getElementById('p1')?.classList.remove('show');
  document.getElementById('p2')?.classList.remove('show');
  
  try{
    // Send requests to both models
    const t0 = Date.now();
    
    const requests = [
      fetch('/api/predict', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({text:s, model:'phobert'})
      }),
      fetch('/api/predict', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({text:s, model:'bilstm'})
      })
    ];
    
    const responses = await Promise.all(requests);
    const lat = Date.now() - t0;
    
    if(!responses[0].ok || !responses[1].ok) throw new Error('HTTP error');
    
    const data1 = await responses[0].json();
    const data2 = await responses[1].json();
    
    if(data1.success && data1.data && data2.success && data2.data){
      fill(data1.data, lat/2, 1);  // PhoBERT on p1
      fill(data2.data, lat/2, 2);  // BiLSTM on p2
      document.getElementById('p1')?.classList.add('show');
      setTimeout(() => document.getElementById('p2')?.classList.add('show'), 100);
    }else{
      showErr(data1.error || data2.error || 'Server error');
    }
  }catch(e){
    showErr(e.message);
  }finally{
    btn.disabled = false;
    btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Predict</span>';
    if(prog)prog.classList.remove('on');
  }
};

function fill(data, lat, panelNum){
  const t = data.tokens || [];
  const e = data.entities || [];
  
  const bodyId = 'body' + panelNum;
  const body = document.getElementById(bodyId);
  if(!body)return;
  
  if(t.length === 0){
    body.innerHTML = '<div class="empty"><div class="eicon">📭</div><p>No tokens</p></div>';
    return;
  }
  
  const map = new Array(t.length).fill(null);
  
  e.forEach(ent => {
    const entTokens = ent.tokens || [];
    if(entTokens.length === 0)return;
    
    for(let start = 0; start <= t.length - entTokens.length; start++){
      let match = true;
      for(let j = 0; j < entTokens.length; j++){
        if(t[start + j] !== entTokens[j]){
          match = false;
          break;
        }
      }
      if(match){
        for(let j = 0; j < entTokens.length; j++){
          if(map[start + j] === null)map[start + j] = ent;
        }
      }
    }
  });
  
  let html = '';
  let i = 0;
  let delay = 0;
  while(i < t.length){
    const ent = map[i];
    if(ent && ent.label){
      const start = i;
      const w = [t[i]];
      let j = i + 1;
      while(j < t.length && map[j] === ent)w.push(t[j++]);
      
      const c = EC[ent.label] || EC.PRODUCT_NAME;
      // Use entity.text if available (for PRICE display like "3,5 triệu"), otherwise use tokens
      const displayText = ent.text || w.join(' ');
      const displayWords = displayText.split(' ').map(x => '<span class="chunk-word">' + esc(x) + '</span>').join('');
      html += '<div class="chunk" style="background:' + c.bg + ';border-color:' + c.border + ';color:' + c.color + ';animation-delay:' + (delay*30) + 'ms"><span class="chunk-lbl">' + esc(ent.label) + '</span><div class="chunk-words">' + displayWords + '</div></div>';
      delay++;
      i = j;
    }else{
      i++;
    }
  }
  
  body.innerHTML = '<div class="tflow">' + html + '</div>';
  
  document.getElementById('tok' + panelNum).textContent = t.length;
  document.getElementById('ent' + panelNum).textContent = e.length;
  document.getElementById('foot' + panelNum).style.display = 'flex';
  document.getElementById('lat' + panelNum).textContent = Math.round(lat) + 'ms';
}

window.clr = function(){
  const txt = document.getElementById('txt');
  if(txt)txt.value = '';
  
  const cc = document.getElementById('cc');
  if(cc)cc.textContent = '0';
  
  // Clear both panels
  [1, 2].forEach(n => {
    const p = document.getElementById('p' + n);
    if(p)p.classList.remove('show');
    
    const body = document.getElementById('body' + n);
    if(body)body.innerHTML = '<div class="empty"><div class="eicon">🤖</div><p>Enter text and click <strong>Predict</strong></p></div>';
    
    const foot = document.getElementById('foot' + n);
    if(foot)foot.style.display = 'none';
    
    const lat = document.getElementById('lat' + n);
    if(lat)lat.textContent = '';
  });
  
  const err = document.getElementById('err');
  if(err)err.innerHTML = '';
};

function showErr(msg){
  const err = document.getElementById('err');
  if(err)err.innerHTML = '<div class="error">⚠️ ' + esc(msg) + '</div>';
  
  document.getElementById('p1')?.classList.remove('show');
  document.getElementById('p2')?.classList.remove('show');
}

window.setEx = function(i){};

// Attach event listeners
(function(){
  const txt = document.getElementById('txt');
  if(!txt)return;
  
  txt.addEventListener('input', function(){
    const cc = document.getElementById('cc');
    if(cc)cc.textContent = this.value.length;
  });
  
  txt.addEventListener('keydown', function(e){
    if(e.key === 'Enter' && (e.ctrlKey || e.metaKey)){
      window.run();
    }
  });
})();
