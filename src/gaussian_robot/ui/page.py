"""Single-page dashboard HTML (config panel + live stream). Served as a string."""

PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>gaussian-robot dashboard</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 14px/1.4 -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px;
         display: grid; grid-template-columns: minmax(280px, 320px) 1fr; gap: 16px; }
  h1 { font-size: 16px; margin: 0 0 8px; }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: #888; margin: 12px 0 6px; }
  .panel { border: 1px solid #444; border-radius: 8px; padding: 12px; background: rgba(127,127,127,.06); }
  label { display: block; margin: 6px 0 2px; font-size: 12px; color: #aaa; }
  input, select { width: 100%; box-sizing: border-box; padding: 4px 6px; border-radius: 4px;
                  border: 1px solid #555; background: transparent; color: inherit; }
  .row { display: flex; gap: 8px; } .row > * { flex: 1; }
  button { margin-top: 10px; padding: 6px 12px; border: 0; border-radius: 6px; cursor: pointer;
           font: inherit; background: #2a7; color: #fff; }
  button.ghost { background: #444; color: inherit; }
  #status { margin-top: 8px; font-size: 12px; color: #9cf; }
  .views { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .views figure { margin: 0; } .views figcaption { font-size: 11px; color: #888; }
  img.panel { width: 100%; border-radius: 6px; background: #000; aspect-ratio: 1/1; object-fit: cover; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
  .chip { padding: 3px 8px; border-radius: 999px; background: #333; font-size: 12px; }
  pre#reason { white-space: pre-wrap; max-height: 140px; overflow: auto; font-size: 11px;
               background: rgba(0,0,0,.3); padding: 8px; border-radius: 6px; }
  pre#vllm-log { white-space: pre-wrap; max-height: 160px; overflow: auto; font-size: 11px;
                 background: rgba(0,0,0,.3); padding: 8px; border-radius: 6px; }
  canvas#world-map { width: 100%; max-width: 460px; aspect-ratio: 1/1; border-radius: 6px;
                     background: #111; display: block; }
  @media (max-width: 900px) {
    body { grid-template-columns: 1fr; }
    .views { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<section class="panel">
  <h1>gaussian-robot</h1>
  <h2>Backends</h2>
  <label>PLY / splat path</label>
  <input id="ply_path" placeholder="data/scene.ply"/>
  <div class="row">
    <div><label>vLLM base URL</label><input id="vlm_base_url"/></div>
    <div><label>vLLM model</label><input id="vlm_model"/></div>
  </div>
  <label><input type="checkbox" id="use_real_vlm" style="width:auto"/> use live vLLM (else demo VLM)</label>
  <label><input type="checkbox" id="start_vllm" style="width:auto"/> launch vLLM before running</label>
  <label><input type="checkbox" id="vlm_enable_thinking" style="width:auto"/> enable thinking mode</label>
  <div class="row">
    <div><label>vLLM bind host</label><input id="vllm_host"/></div>
    <div><label>vLLM port</label><input id="vllm_port" type="number"/></div>
  </div>
  <label>vLLM extra args</label>
  <input id="vllm_extra_args" placeholder="--dtype auto --gpu-memory-utilization 0.9"/>
  <label>CUDA device (renderer)</label>
  <input id="cuda_device" placeholder="cuda:0"/>
  <div class="row">
    <button onclick="startVLLM()" class="ghost">Start vLLM</button>
    <button onclick="stopVLLM()" class="ghost">Stop vLLM</button>
    <button onclick="refreshVLLM()" class="ghost">vLLM status</button>
  </div>
  <pre id="vllm-log">vLLM log idle</pre>
  <h2>Task</h2>
  <label>task prompt (optional)</label>
  <input id="task_prompt" placeholder="e.g. find the office door"/>
  <h2>Scene &amp; exploration</h2>
  <label>up axis</label>
  <select id="up_axis"><option>auto</option><option>y</option><option>-y</option><option>x</option><option>-x</option><option>z</option><option>-z</option></select>
  <div class="row">
    <div><label>bounds min (x,y,z)</label><input id="bounds_min"/></div>
    <div><label>bounds max (x,y,z)</label><input id="bounds_max"/></div>
  </div>
  <div class="row">
    <div><label>step fraction</label><input id="action_step_fraction" type="number" step="0.005"/></div>
    <div><label>coverage radius</label><input id="coverage_radius" type="number" step="0.1" placeholder="auto"/></div>
  </div>
  <div class="row">
    <div><label>max steps / walk</label><input id="max_steps" type="number"/></div>
    <div><label>pose budget</label><input id="pose_budget" type="number"/></div>
    <div><label># seeds</label><input id="num_seeds" type="number"/></div>
    <div><label>pose target</label><input id="pose_target" type="number"/></div>
  </div>
  <div class="row">
    <button onclick="saveConfig()" class="ghost">Save config</button>
    <button onclick="loadScene()" class="ghost">Load scene</button>
  </div>
  <div class="row">
    <button onclick="run()">Run session</button>
    <button onclick="runStepping()" class="ghost">Step-by-step</button>
    <button id="btn-next" onclick="nextStep()" class="ghost" disabled>Next step</button>
    <button onclick="testForward()" class="ghost">Test forward</button>
  </div>
  <div id="status"></div>
</section>

<section class="panel">
  <div class="chips" id="chips"><span class="chip">idle</span></div>
  <div class="views" style="grid-template-columns:repeat(4,1fr);">
    <figure><figcaption>rgb view</figcaption><img id="rgb" class="panel"/></figure>
    <figure><figcaption>depth</figcaption><img id="depth" class="panel"/></figure>
    <figure><figcaption>opacity (holes = dark)</figcaption><img id="confidence" class="panel"/></figure>
    <figure><figcaption>map (body-fixed)</figcaption><img id="body-map" class="panel"/></figure>
  </div>
  <h2>Global coverage (world frame)</h2>
  <canvas id="world-map" width="600" height="600"></canvas>
  <h2>Walk replay (interpolated fly-through)</h2>
  <div class="row">
    <select id="movie-walk" style="flex:2"></select>
    <div style="flex:1"><label style="margin:0">frames/step</label><input id="movie-per" type="number" value="8" min="1"/></div>
    <button onclick="buildMovie()" class="ghost" style="align-self:end">Build</button>
  </div>
  <img id="movie-frame" class="panel" style="max-width:512px;margin-top:6px"/>
  <div id="movie-cap" style="min-height:34px;font-size:12px;color:#cdd;white-space:pre-wrap;
       background:rgba(0,0,0,.3);padding:6px 8px;border-radius:6px;margin-top:6px">—</div>
  <div class="row" style="align-items:center;margin-top:6px">
    <button id="movie-play" onclick="toggleMovie()" class="ghost" disabled style="flex:0 0 auto">▶ Play</button>
    <input type="range" id="movie-scrub" min="0" max="0" value="0" oninput="scrubMovie()" style="flex:3"/>
    <span id="movie-label" style="font-size:12px;color:#aaa;min-width:64px;flex:0 0 auto">—</span>
  </div>
  <h2>Scene description</h2>
  <pre id="scene-desc" style="white-space:pre-wrap;max-height:140px;overflow:auto;font-size:12px;
       background:rgba(80,140,80,.12);padding:8px;border-radius:6px;">—</pre>
  <h2>Action log</h2>
  <pre id="action-log" style="white-space:pre-wrap;max-height:120px;overflow:auto;font-size:11px;
       background:rgba(0,0,0,.3);padding:8px;border-radius:6px;">—</pre>
  <h2>VLM decision / reasoning</h2>
  <pre id="reason">—</pre>
</section>

<script>
let BOUNDS = null, UP = "y";
const num = (v, d) => (v === "" || v === null || v === undefined ? d : Number(v));
function floorAxes(up){ const u={x:0,y:1,z:2}[(up||"y").replace("-","")]; return [0,1,2].filter(i=>i!==u); }
const AXIS_NAME = i => ["X","Y","Z"][i];

async function loadConfig(){
  const c = await fetch("/api/config").then(r=>r.json());
  for (const k of ["ply_path","vlm_base_url","vlm_model","vllm_host","task_prompt","cuda_device"]) document.getElementById(k).value = c[k] ?? "";
  document.getElementById("up_axis").value = c.up_axis;
  document.getElementById("use_real_vlm").checked = !!c.use_real_vlm;
  document.getElementById("start_vllm").checked = !!c.start_vllm;
  document.getElementById("vlm_enable_thinking").checked = !!c.vlm_enable_thinking;
  document.getElementById("vllm_port").value = c.vllm_port;
  document.getElementById("vllm_extra_args").value = (c.vllm_extra_args||[]).join(" ");
  document.getElementById("bounds_min").value = (c.bounds_min||[0,0,0]).join(",");
  document.getElementById("bounds_max").value = (c.bounds_max||[10,10,10]).join(",");
  document.getElementById("action_step_fraction").value = c.action_step_fraction;
  document.getElementById("coverage_radius").value = c.coverage_radius ?? "";
  document.getElementById("max_steps").value = c.max_steps;
  document.getElementById("pose_budget").value = c.pose_budget;
  document.getElementById("num_seeds").value = c.num_seeds;
  document.getElementById("pose_target").value = c.pose_target;
}
function gather(){
  const split3 = v => v.split(",").map(parseFloat);
  const splitArgs = v => v.trim() === "" ? [] : v.trim().split(/\s+/);
  const cr = document.getElementById("coverage_radius").value;
  return {
    task_prompt: document.getElementById("task_prompt").value || "",
    ply_path: document.getElementById("ply_path").value || null,
    vlm_base_url: document.getElementById("vlm_base_url").value,
    vlm_model: document.getElementById("vlm_model").value,
    use_real_vlm: document.getElementById("use_real_vlm").checked,
    start_vllm: document.getElementById("start_vllm").checked,
    vlm_enable_thinking: document.getElementById("vlm_enable_thinking").checked,
    vllm_host: document.getElementById("vllm_host").value,
    vllm_port: num(document.getElementById("vllm_port").value, 8000),
    vllm_extra_args: splitArgs(document.getElementById("vllm_extra_args").value),
    cuda_device: document.getElementById("cuda_device").value || "cuda:0",
    up_axis: document.getElementById("up_axis").value,
    bounds_min: split3(document.getElementById("bounds_min").value),
    bounds_max: split3(document.getElementById("bounds_max").value),
    action_step_fraction: num(document.getElementById("action_step_fraction").value, 0.03),
    coverage_radius: cr === "" ? null : Number(cr),
    max_steps: num(document.getElementById("max_steps").value, 40),
    pose_budget: num(document.getElementById("pose_budget").value, 200),
    num_seeds: num(document.getElementById("num_seeds").value, 5),
    pose_target: num(document.getElementById("pose_target").value, 30),
  };
}
async function saveConfig(){
  const r = await fetch("/api/config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(gather())});
  setStatus("saved: " + (r.ok ? "ok" : "failed"));
}
async function startVLLM(){
  await saveConfig();
  setStatus("starting vLLM...");
  const r = await fetch("/api/vllm/start",{method:"POST"}).then(r=>r.json());
  showVLLM(r);
}
async function stopVLLM(){
  const r = await fetch("/api/vllm/stop",{method:"POST"}).then(r=>r.json());
  showVLLM(r);
}
async function refreshVLLM(){
  const r = await fetch("/api/vllm/status").then(r=>r.json());
  showVLLM(r);
}
function showVLLM(r){
  const state = r.ready ? "ready" : (r.running ? "starting" : "stopped");
  const pid = r.pid ? ` pid ${r.pid}` : "";
  const code = r.returncode === null || r.returncode === undefined ? "" : ` exit ${r.returncode}`;
  const path = r.log_path ? `\n\nlog: ${r.log_path}` : "";
  document.getElementById("vllm-log").textContent = (r.log_tail || "no vLLM log yet") + path;
  setStatus(`vLLM ${state}${pid}${code}`);
}
function setStatus(t){ document.getElementById("status").textContent = t; }
function setChips(o){ document.getElementById("chips").innerHTML =
  Object.entries(o).map(([k,v])=>`<span class="chip"><b>${k}</b> ${v}</span>`).join(""); }

let SEEDS = [], SEED_KINDS = [], MARKS = [], FRONTIERS = [], GAPS = [], LAST_STEP = null;
const seedColor = kind => (kind === "capture" ? "#f5be28" : "#e0662a");  // amber=real, orange=synthetic
const walkIndex = wid => { const n = String(wid).match(/\\d+/); return n ? parseInt(n[0], 10) : -1; };
const walkKind = wid => SEED_KINDS[walkIndex(wid)] || "?";
function drawWorld(ctx, sampled, trail, pose){
  if(!BOUNDS) return;
  const [a,b] = floorAxes(UP);
  const lo=[BOUNDS.min[a],BOUNDS.min[b]], hi=[BOUNDS.max[a],BOUNDS.max[b]];
  const W=ctx.canvas.width, H=ctx.canvas.height, pad=10;
  const tx = x => pad + (x-lo[0])/(hi[0]-lo[0])*(W-2*pad);
  const ty = y => pad + (1-(y-lo[1])/(hi[1]-lo[1]))*(H-2*pad);
  ctx.fillStyle="#181818"; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle="#666"; ctx.lineWidth=1;
  ctx.strokeRect(tx(lo[0]),ty(hi[1]),tx(hi[0])-tx(lo[0]),ty(lo[1])-ty(hi[1]));
  // reconstruction frontiers (static gaps to fill): faint cyan squares, drawn underneath
  ctx.fillStyle="rgba(80,200,210,0.45)";
  (FRONTIERS||[]).forEach(p=>{ ctx.fillRect(tx(p[0])-1.5, ty(p[1])-1.5, 3, 3); });
  // Tier-3 3D coverage gaps (occupied-but-unseen voxels: roofs/floors/behind-buildings),
  // floor-projected. Hot-pink hollow squares — the aerial survey targets these.
  ctx.strokeStyle="rgba(255,79,163,0.85)"; ctx.lineWidth=1.5;
  (GAPS||[]).forEach(p=>{ ctx.strokeRect(tx(p[0])-2.5, ty(p[1])-2.5, 5, 5); });
  ctx.fillStyle="#285ad0";
  (sampled||[]).forEach(p=>{ ctx.beginPath(); ctx.arc(tx(p[0]),ty(p[1]),2,0,7); ctx.fill(); });
  // seeds: where walks start from. Amber ring = real capture pose, orange = synthetic fallback.
  ctx.lineWidth=2; ctx.font="9px sans-serif";
  (SEEDS||[]).forEach((p,i)=>{ const x=tx(p[0]),y=ty(p[1]); const c=seedColor(SEED_KINDS[i]);
    ctx.strokeStyle=c; ctx.beginPath(); ctx.arc(x,y,6,0,7); ctx.stroke();
    ctx.fillStyle=c; ctx.fillText(i, x+7, y+3); });
  if(trail && trail.length>1){
    ctx.strokeStyle="#2aa846"; ctx.lineWidth=2; ctx.beginPath();
    trail.forEach((p,i)=>{ const x=tx(p[0]),y=ty(p[1]); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke();
  }
  // marked fill-in poses (the deliverable): purple diamonds, drawn on top
  ctx.fillStyle="#c050ff";
  (MARKS||[]).forEach(p=>{ const x=tx(p[0]),y=ty(p[1]),s=5;
    ctx.beginPath(); ctx.moveTo(x,y-s); ctx.lineTo(x+s,y); ctx.lineTo(x,y+s); ctx.lineTo(x-s,y);
    ctx.closePath(); ctx.fill(); });
  if(pose){ ctx.fillStyle="#d6201e"; ctx.beginPath(); ctx.arc(tx(pose[a]),ty(pose[b]),4,0,7); ctx.fill(); }
  // legend
  ctx.font="11px sans-serif"; ctx.textBaseline="middle";
  const items=[["#50c8d2","gap (frontier)"],["#ff4fa3","gap (3D: roofs/floors)"],["#285ad0","visited"],["#f5be28","seed (capture)"],["#e0662a","seed (fallback)"],["#2aa846","current trail"],["#d6201e","robot"],["#c050ff","marked (fill-in)"]];
  items.forEach(([c,t],i)=>{ const y=14+i*16; ctx.fillStyle=c;
    ctx.beginPath(); ctx.arc(14,y,4,0,7); ctx.fill(); ctx.fillStyle="#bbb"; ctx.fillText(t,24,y); });
  // world axis labels: horizontal = +floor axis a, vertical = +floor axis b (up axis is out of plane)
  ctx.fillStyle="#9aa"; ctx.font="12px sans-serif";
  ctx.fillText("+"+AXIS_NAME(a)+" \\u2192", W-58, H-12);
  ctx.fillText("up: "+UP, W-58, 16);
  ctx.save(); ctx.translate(14, H-14); ctx.rotate(-Math.PI/2);
  ctx.fillText("+"+AXIS_NAME(b)+" \\u2192", 0, 0); ctx.restore();
  ctx.textBaseline="alphabetic";
}

let rgbTimer=null;
function playRgb(frames){
  if(rgbTimer){ clearInterval(rgbTimer); rgbTimer=null; }
  const img=document.getElementById("rgb");
  if(!frames.length) return;
  if(frames.length===1){ img.src=frames[0]; return; }
  let i=0;
  rgbTimer=setInterval(()=>{ img.src=frames[i++];
    if(i>=frames.length){ clearInterval(rgbTimer); rgbTimer=null; } }, 45);
}
let animTimer=null;
async function testForward(){
  if(animTimer){ clearInterval(animTimer); animTimer=null; }
  setStatus("rendering forward animation...");
  try {
    const r = await fetch("/api/animate-forward").then(r=>r.json());
    if(!r.ok){ setStatus("animate failed: "+(r.error||"unknown")); return; }
    const frames = r.frames;
    let i = 0;
    setStatus(`forward animation: ${r.n_steps} steps, step_size=${r.step_size.toFixed(2)}`);
    animTimer = setInterval(()=>{
      document.getElementById("rgb").src = frames[i];
      setChips({animation:`frame ${i+1}/${frames.length}`, step_size:r.step_size.toFixed(2)});
      i++;
      if(i >= frames.length){ i=0; }
    }, 80);
  } catch(e){ setStatus("animate error: "+e.message); }
}

async function loadScene(){
  await saveConfig();
  setStatus("loading scene...");
  try {
    const r = await fetch("/api/load").then(r=>r.json());
    if(!r.ok){ setStatus("load failed: "+(r.error||"unknown")); return; }
    document.getElementById("rgb").src = r.rgb;
    if(r.depth) document.getElementById("depth").src = r.depth;
    setChips({status:"loaded", renderer:r.renderer});
    document.getElementById("bounds_min").value = r.bounds_min.join(",");
    document.getElementById("bounds_max").value = r.bounds_max.join(",");
    setStatus("scene loaded ("+r.renderer+")");
  } catch(e){ setStatus("load error: "+e.message); }
}

let stream=null, stepping=false;
async function nextStep(){
  await fetch("/api/step/next",{method:"POST"});
}
function runStepping(){ stepping=true; run("/api/run/stepping"); }
async function run(endpoint){
  endpoint = endpoint || "/api/run";
  if(stream){ stream.close(); stream=null; }
  await saveConfig();
  setStatus("connecting..."); setChips({status:"connecting"});
  document.getElementById("btn-next").disabled = !stepping;
  stream = new EventSource(endpoint);
  const vlmMode = document.getElementById("use_real_vlm").checked ? "live VLM" : "demo VLM";
  stream.onopen = ()=>setStatus("running ("+vlmMode+")");
  stream.onmessage = (e)=>{
    const m = JSON.parse(e.data);
    if(m.type==="session_start"){ BOUNDS={min:m.bounds_min,max:m.bounds_max}; UP=m.up_axis;
      SEEDS = m.seeds || []; SEED_KINDS = m.seed_kinds || []; MARKS = []; LAST_STEP = null;
      FRONTIERS = m.frontiers || []; GAPS = m.gaps || [];
      document.getElementById("action-log").textContent="";
      document.getElementById("scene-desc").textContent="—";
      resetMovies();
      drawWorld(document.getElementById("world-map").getContext("2d"), [], [], null);
      const seedsLabel = (m.requested_seeds && m.requested_seeds !== m.total_seeds)
        ? `${m.total_seeds}/${m.requested_seeds}` : `${m.total_seeds}`;
      setChips({status:"running", vlm:vlmMode, seeds:seedsLabel, up:UP}); return; }
    if(m.type==="walk_end"){
      const log = document.getElementById("action-log");
      log.textContent += `── ${m.walk_id} (${walkKind(m.walk_id)}) ended: ${m.reason} (${m.steps} steps) ──\n`;
      log.scrollTop = log.scrollHeight; addWalkOption(m.walk_id); return; }
    if(m.type==="mark"){
      MARKS.push(m.floor);
      const tgt = num(document.getElementById("pose_target").value, 30);
      const how = m.auto ? "auto" : "vlm";
      const log = document.getElementById("action-log");
      log.textContent += `★ ${m.walk_id} #${m.step}: marked fill-in pose [${how}] (${m.count}/${tgt})\n`;
      log.scrollTop = log.scrollHeight;
      setChips({walk:`${m.walk_id} (${walkKind(m.walk_id)})`, action:`mark:${how}`, marks:`${m.count}/${tgt}`});
      const ctx=document.getElementById("world-map").getContext("2d");
      if(LAST_STEP) drawWorld(ctx, LAST_STEP.sampled, LAST_STEP.trail, LAST_STEP.pose);
      else drawWorld(ctx, [], [], null);
      return; }
    if(m.type==="scene_describe"){
      document.getElementById("scene-desc").textContent = m.description || "—";
      const log = document.getElementById("action-log");
      log.textContent += `[${m.walk_id} #${m.step}] describe: ${m.description}\n`;
      log.scrollTop = log.scrollHeight;
      setChips({walk:`${m.walk_id} (${walkKind(m.walk_id)})`, step:m.step, action:"describe"});
      return; }
    if(m.type==="session_end"){ setStatus("ended: "+m.reason); setChips({status:"ended",reason:m.reason,
      steps:m.total_steps, poses:m.total_poses}); stream.close(); stream=null; stepping=false;
      document.getElementById("btn-next").disabled=true; return; }
    if(m.type==="error"){ setStatus("error: "+m.message); setChips({status:"error"}); stream.close(); stream=null;
      stepping=false; document.getElementById("btn-next").disabled=true; return; }
    if(m.type==="step"){
      if(m.panels.rgb) playRgb([...(m.tween||[]), m.panels.rgb]);
      if(m.panels.depth) document.getElementById("depth").src=m.panels.depth;
      if(m.panels.confidence) document.getElementById("confidence").src=m.panels.confidence;
      if(m.panels.map) document.getElementById("body-map").src=m.panels.map;
      addWalkOption(m.walk_id);
      document.getElementById("reason").textContent = m.raw_text || m.action;
      const log = document.getElementById("action-log");
      const pos = m.pose.map(v=>v.toFixed(1)).join(",");
      const flag = m.blocked ? " [blocked: wall ahead]" : (m.degenerate ? " [degenerate]" : "");
      log.textContent += `[${m.walk_id} #${m.step}] ${m.action} → (${pos})  nov=${m.novelty.toFixed(2)}${flag}\n`;
      log.scrollTop = log.scrollHeight;
      setChips({walk:`${m.walk_id} (${walkKind(m.walk_id)})`, step:`${m.step}/${m.budget}`, action:m.action,
        novelty:m.novelty.toFixed(2), blocked:m.blocked?"!":"", degenerate:m.degenerate?"!":"",
        "cov(floor)":(100*m.coverage_floor).toFixed(1)+"%",
        "cov(pose)":(100*m.coverage_pose_space).toFixed(1)+"%"});
      LAST_STEP = m;
      drawWorld(document.getElementById("world-map").getContext("2d"), m.sampled, m.trail, m.pose);
    }
  };
  stream.onerror = ()=>{ setStatus("disconnected"); stepping=false; document.getElementById("btn-next").disabled=true; };
}

// ---- Walk replay (interpolated fly-through) ----
let MOVIE = {frames:[], captions:[], i:0, timer:null};
function addWalkOption(wid){
  const sel=document.getElementById("movie-walk");
  if([...sel.options].some(o=>o.value===wid)) return;
  const o=document.createElement("option"); o.value=wid; o.textContent=wid; sel.appendChild(o);
}
function resetMovies(){
  stopMovie();
  document.getElementById("movie-walk").innerHTML="";
  MOVIE={frames:[], captions:[], i:0, timer:null};
  const scrub=document.getElementById("movie-scrub"); scrub.max=0; scrub.value=0;
  document.getElementById("movie-label").textContent="—";
  document.getElementById("movie-cap").textContent="—";
  document.getElementById("movie-play").disabled=true;
  document.getElementById("movie-frame").removeAttribute("src");
}
async function buildMovie(){
  const wid=document.getElementById("movie-walk").value;
  if(!wid){ setStatus("select a walk first"); return; }
  const per=num(document.getElementById("movie-per").value, 8);
  setStatus("rendering fly-through for "+wid+"…");
  try{
    const r=await fetch(`/api/walk-movie?walk=${encodeURIComponent(wid)}&per=${per}`).then(r=>r.json());
    if(!r.ok){ setStatus("movie failed: "+(r.error||"unknown")); return; }
    MOVIE.frames=r.frames||[]; MOVIE.captions=r.captions||[]; MOVIE.i=0;
    const scrub=document.getElementById("movie-scrub");
    scrub.max=Math.max(0, MOVIE.frames.length-1); scrub.value=0;
    document.getElementById("movie-play").disabled = MOVIE.frames.length<2;
    showMovieFrame(0);
    setStatus(`movie ${wid}: ${MOVIE.frames.length} frames from ${r.checkpoints} checkpoints`);
  }catch(e){ setStatus("movie error: "+e.message); }
}
function showMovieFrame(i){
  if(!MOVIE.frames.length) return;
  MOVIE.i=Math.max(0, Math.min(i, MOVIE.frames.length-1));
  document.getElementById("movie-frame").src=MOVIE.frames[MOVIE.i];
  document.getElementById("movie-scrub").value=MOVIE.i;
  document.getElementById("movie-label").textContent=`${MOVIE.i+1}/${MOVIE.frames.length}`;
  document.getElementById("movie-cap").textContent=MOVIE.captions[MOVIE.i] || "—";
}
function scrubMovie(){ stopMovie(); showMovieFrame(+document.getElementById("movie-scrub").value); }
function stopMovie(){ if(MOVIE.timer){ clearInterval(MOVIE.timer); MOVIE.timer=null; }
  document.getElementById("movie-play").textContent="▶ Play"; }
function toggleMovie(){
  if(MOVIE.timer){ stopMovie(); return; }
  if(MOVIE.frames.length<2) return;
  document.getElementById("movie-play").textContent="⏸ Pause";
  MOVIE.timer=setInterval(()=>{ showMovieFrame(MOVIE.i+1>=MOVIE.frames.length ? 0 : MOVIE.i+1); }, 80);
}

loadConfig();
</script>
</body>
</html>
"""
