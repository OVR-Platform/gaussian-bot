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
  canvas#world-map { width: 100%; aspect-ratio: 1/1; border-radius: 6px; background: #111; }
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
  <div class="row">
    <div><label>vLLM bind host</label><input id="vllm_host"/></div>
    <div><label>vLLM port</label><input id="vllm_port" type="number"/></div>
  </div>
  <label>vLLM extra args</label>
  <input id="vllm_extra_args" placeholder="--dtype auto --gpu-memory-utilization 0.9"/>
  <div class="row">
    <button onclick="startVLLM()" class="ghost">Start vLLM</button>
    <button onclick="stopVLLM()" class="ghost">Stop vLLM</button>
    <button onclick="refreshVLLM()" class="ghost">vLLM status</button>
  </div>
  <pre id="vllm-log">vLLM log idle</pre>
  <h2>Scene &amp; exploration</h2>
  <label>up axis</label>
  <select id="up_axis"><option>y</option><option>x</option><option>z</option></select>
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
  </div>
  <div class="row">
    <button onclick="saveConfig()" class="ghost">Save config</button>
    <button onclick="run()">Run session</button>
  </div>
  <div id="status"></div>
</section>

<section class="panel">
  <div class="chips" id="chips"><span class="chip">idle</span></div>
  <div class="views">
    <figure><figcaption>rgb view</figcaption><img id="rgb" class="panel"/></figure>
    <figure><figcaption>depth</figcaption><img id="depth" class="panel"/></figure>
    <figure><figcaption>map (body-fixed)</figcaption><img id="body-map" class="panel"/></figure>
  </div>
  <h2>Global coverage (world frame)</h2>
  <canvas id="world-map" width="400" height="400"></canvas>
  <h2>VLM decision / reasoning</h2>
  <pre id="reason">—</pre>
</section>

<script>
let BOUNDS = null, UP = "y";
const num = (v, d) => (v === "" || v === null || v === undefined ? d : Number(v));
function floorAxes(up){ const all=[0,1,2]; const u={x:0,y:1,z:2}[up]; return all.filter(i=>i!==u); }

async function loadConfig(){
  const c = await fetch("/api/config").then(r=>r.json());
  for (const k of ["ply_path","vlm_base_url","vlm_model","vllm_host"]) document.getElementById(k).value = c[k] ?? "";
  document.getElementById("up_axis").value = c.up_axis;
  document.getElementById("use_real_vlm").checked = !!c.use_real_vlm;
  document.getElementById("start_vllm").checked = !!c.start_vllm;
  document.getElementById("vllm_port").value = c.vllm_port;
  document.getElementById("vllm_extra_args").value = (c.vllm_extra_args||[]).join(" ");
  document.getElementById("bounds_min").value = (c.bounds_min||[0,0,0]).join(",");
  document.getElementById("bounds_max").value = (c.bounds_max||[10,10,10]).join(",");
  document.getElementById("action_step_fraction").value = c.action_step_fraction;
  document.getElementById("coverage_radius").value = c.coverage_radius ?? "";
  document.getElementById("max_steps").value = c.max_steps;
  document.getElementById("pose_budget").value = c.pose_budget;
  document.getElementById("num_seeds").value = c.num_seeds;
}
function gather(){
  const split3 = v => v.split(",").map(parseFloat);
  const splitArgs = v => v.trim() === "" ? [] : v.trim().split(/\s+/);
  const cr = document.getElementById("coverage_radius").value;
  return {
    ply_path: document.getElementById("ply_path").value || null,
    vlm_base_url: document.getElementById("vlm_base_url").value,
    vlm_model: document.getElementById("vlm_model").value,
    use_real_vlm: document.getElementById("use_real_vlm").checked,
    start_vllm: document.getElementById("start_vllm").checked,
    vllm_host: document.getElementById("vllm_host").value,
    vllm_port: num(document.getElementById("vllm_port").value, 8000),
    vllm_extra_args: splitArgs(document.getElementById("vllm_extra_args").value),
    up_axis: document.getElementById("up_axis").value,
    bounds_min: split3(document.getElementById("bounds_min").value),
    bounds_max: split3(document.getElementById("bounds_max").value),
    action_step_fraction: num(document.getElementById("action_step_fraction").value, 0.03),
    coverage_radius: cr === "" ? null : Number(cr),
    max_steps: num(document.getElementById("max_steps").value, 40),
    pose_budget: num(document.getElementById("pose_budget").value, 200),
    num_seeds: num(document.getElementById("num_seeds").value, 5),
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

function drawWorld(ctx, sampled, trail, pose){
  if(!BOUNDS) return;
  const [a,b] = floorAxes(UP);
  const lo=[BOUNDS.min[a],BOUNDS.min[b]], hi=[BOUNDS.max[a],BOUNDS.max[b]];
  const W=ctx.canvas.width, H=ctx.canvas.height, pad=10;
  const tx = x => pad + (x-lo[0])/(hi[0]-lo[0])*(W-2*pad);
  const ty = y => pad + (1-(y-lo[1])/(hi[1]-lo[1]))*(H-2*pad);
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle="#444"; ctx.strokeRect(tx(lo[0]),ty(hi[1]),tx(hi[0])-tx(lo[0]),ty(lo[1])-ty(hi[1]));
  ctx.fillStyle="#285ad0";
  (sampled||[]).forEach(p=>{ ctx.beginPath(); ctx.arc(tx(p[a]),ty(p[b]),2,0,7); ctx.fill(); });
  if(trail && trail.length>1){
    ctx.strokeStyle="#2aa846"; ctx.lineWidth=2; ctx.beginPath();
    trail.forEach((p,i)=>{ const x=tx(p[a]),y=ty(p[b]); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke();
  }
  if(pose){ ctx.fillStyle="#d6201e"; ctx.beginPath(); ctx.arc(tx(pose[a]),ty(pose[b]),4,0,7); ctx.fill(); }
}

let stream=null;
async function run(){
  if(stream){ stream.close(); stream=null; }
  await saveConfig();
  setStatus("connecting..."); setChips({status:"connecting"});
  stream = new EventSource("/api/run");
  stream.onopen = ()=>setStatus("running");
  stream.onmessage = (e)=>{
    const m = JSON.parse(e.data);
    if(m.type==="session_start"){ BOUNDS={min:m.bounds_min,max:m.bounds_max}; UP=m.up_axis;
      setChips({status:"running", seeds:m.total_seeds, up:UP}); return; }
    if(m.type==="session_end"){ setStatus("ended: "+m.reason); setChips({status:"ended",reason:m.reason,
      steps:m.total_steps, poses:m.total_poses}); stream.close(); stream=null; return; }
    if(m.type==="error"){ setStatus("error: "+m.message); setChips({status:"error"}); stream.close(); stream=null; return; }
    if(m.type==="step"){
      if(m.panels.rgb) document.getElementById("rgb").src=m.panels.rgb;
      if(m.panels.depth) document.getElementById("depth").src=m.panels.depth;
      if(m.panels.map) document.getElementById("body-map").src=m.panels.map;
      document.getElementById("reason").textContent = m.raw_text || m.action;
      setChips({seed:m.seed_id, step:`${m.step}/${m.budget}`, action:m.action,
        novelty:m.novelty.toFixed(2), degenerate:m.degenerate?"!":"",
        "cov(floor)":(100*m.coverage_floor).toFixed(1)+"%",
        "cov(pose)":(100*m.coverage_pose_space).toFixed(1)+"%"});
      drawWorld(document.getElementById("world-map").getContext("2d"), m.sampled, m.trail, m.pose);
    }
  };
  stream.onerror = ()=>setStatus("disconnected");
}
loadConfig();
</script>
</body>
</html>
"""
