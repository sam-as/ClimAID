/* ======================================================
ClimAID Assistant chat page
Talks to /assistant/* (climaid/browser_ui/assistant_api.py). Replies are plain text from the
rule-based assistant; this file only displays them.
====================================================== */

const SESSION_STORAGE_KEY = "climaid_session_id";      // shared with the dashboard (browser.js)
let sessionId = localStorage.getItem(SESSION_STORAGE_KEY);
if(!sessionId){
    sessionId = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + "-" + Math.random().toString(16).slice(2));
    localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
}

const chat = document.getElementById("chat");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const busyBox = document.getElementById("chat-busy");
const busyText = document.getElementById("chat-busy-text");
const chips = document.getElementById("chat-chips");
const fileInput = document.getElementById("chat-file");
const attachBtn = document.getElementById("attach-btn");
const attachMenu = document.getElementById("attach-menu");

let shown = 0;              // messages already displayed
let busy = false;
let busySince = null;
let pollTimer = null;
let uploadKind = "disease";
let lastAssistantText = "";

function api(url, options={}){
    const opts = Object.assign({}, options);
    opts.headers = Object.assign({"X-ClimAID-Session": sessionId}, options.headers || {});
    return fetch(url, opts);
}

/* ---------------- rendering ---------------- */
function escapeHtml(s){
    return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function linkify(escaped){
    return escaped.replace(/(https?:\/\/[^\s<]+|file:\/\/[^\s<]+|\/(?:reports|documentation)\/[^\s<]*)/g, (m)=>{
        let url = m, tail = "";
        while(/[.,;:)\]]$/.test(url)){ tail = url.slice(-1) + tail; url = url.slice(0, -1); }
        const label = url.startsWith("/reports/") ? "Open the report"
                    : url.startsWith("/documentation/") ? "Open in the documentation" : url;
        return `<a href="${url}" target="_blank" rel="noopener">${label}</a>${tail}`;
    });
}

function renderText(text){
    // Blocks: paragraphs, "- item" lists, and indented lines (tables) that keep their spacing.
    let html = "", kind = null, buf = [];
    const flush = ()=>{
        if(!buf.length) return;
        const items = buf.map(l=>linkify(escapeHtml(l)));
        if(kind === "list") html += `<ul>${items.map(l=>`<li>${l}</li>`).join("")}</ul>`;
        else if(kind === "pre") html += `<pre>${items.join("\n")}</pre>`;
        else html += `<p>${items.join("<br>")}</p>`;
        buf = [];
    };
    for(const line of text.split("\n")){
        const bullet = line.match(/^\s*-\s+(.*)$/);
        const k = bullet ? "list" : /^\s{2,}\S/.test(line) ? "pre" : line.trim() === "" ? "blank" : "para";
        if(k !== kind){ flush(); kind = k; }
        if(k === "list") buf.push(bullet[1]);
        else if(k !== "blank") buf.push(line);
    }
    flush();
    return html;
}

function addBubble(msg){
    const div = document.createElement("div");
    div.className = "chat-msg " + (msg.role === "user" ? "from-user" : "from-assistant");
    if(msg.role === "user"){
        div.textContent = msg.text;
    }else{
        div.innerHTML = renderText(msg.text);
        lastAssistantText = msg.text;
    }
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

/* ---------------- suggestions ---------------- */
function setChips(list){
    chips.innerHTML = "";
    for(const [label, text] of list){
        const b = document.createElement("button");
        b.type = "button"; b.className = "chat-chip"; b.textContent = label;
        b.onclick = ()=> send(text);
        chips.appendChild(b);
    }
}

function updateChips(){
    if(busy){ setChips([]); return; }
    const t = lastAssistantText;
    const numbered = [...t.matchAll(/^\s+(\d+)\.\s+(.+)$/gm)];
    if(/Shall I run it\?/.test(t)){
        setChips([["Yes, run it","yes"],["No","no"],["Quick tuning","quick tuning"],["Show settings","settings"]]);
    }else if(numbered.length){
        const short = (x)=> x.length > 30 ? x.slice(0, 28).trim() + "…" : x;
        setChips(numbered.slice(0, 24).map(m=>[`${m[1]}. ${short(m[2])}`, m[1]]));
    }else if(/Say "more detail"/.test(t)){
        setChips([["More detail","more detail"],["All methods","methods"],["Help","help"]]);
    }else if(/Which disease is it/.test(t)){
        setChips([["Dengue","dengue"],["Malaria","malaria"],["Skip","skip"]]);
    }else if(/Trust rating:/.test(t) || /You can now ask/.test(t)){
        setChips([["How reliable is this?","how reliable is this?"],["How was this made?","how was this forecast made?"],
                  ["Open the report","open the report"],
                  ["What does likely range mean?","what does likely range mean?"],
                  ["Climate outlook to 2050","what could happen by 2050?"]]);
    }else{
        setChips([["Forecast the next 12 months","I want a forecast for the next 12 months"],
                  ["Climate outlook to 2050","what could happen by 2050 under different emissions?"],
                  ["What data do I need?","what format should my data be in?"],["Explain the methods","methods"],
                  ["What is WIS?","what is WIS?"],["Help","help"]]);
    }
}

/* ---------------- busy state & polling ---------------- */
function setBusy(on){
    busy = on;
    busyBox.hidden = !on;
    sendBtn.disabled = on;
    attachBtn.disabled = on;
    if(on && !busySince) busySince = Date.now();
    if(!on) busySince = null;
    updateChips();
}

function tickBusy(){
    if(!busy || !busySince) return;
    const s = Math.round((Date.now() - busySince) / 1000);
    const running = /Running the/.test(lastAssistantText);
    const time = s < 60 ? `${s} s` : `${Math.floor(s/60)} min ${s%60} s`;
    busyText.textContent = running
        ? `Running ClimAID… ${time} so far. This can take several minutes; you can leave this page open.`
        : `Working… ${time}`;
}

async function poll(){
    try{
        const r = await api(`/assistant/messages?after=${shown}`);
        if(!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        for(const m of data.messages){ addBubble(m); }
        shown += data.messages.length;
        setBusy(data.busy);
        tickBusy();
    }catch(err){
        busyText.textContent = "Lost contact with the ClimAID server. Is `climaid browse` still running?";
        busyBox.hidden = false;
    }finally{
        clearTimeout(pollTimer);
        pollTimer = setTimeout(poll, busy ? 1500 : 4000);
    }
}

/* ---------------- sending ---------------- */
async function send(text){
    text = (text || "").trim();
    if(!text || busy) return;
    input.value = "";
    setBusy(true);
    try{
        const r = await api("/assistant/message", {method:"POST", headers:{"Content-Type":"application/json"},
                                                  body: JSON.stringify({text})});
        if(!r.ok){
            const e = await r.json().catch(()=>({}));
            addBubble({role:"assistant", text: e.detail || `The server refused the message (HTTP ${r.status}).`});
        }
    }catch(err){
        addBubble({role:"assistant", text:"Could not reach the ClimAID server."});
    }
    poll();
}

document.getElementById("chat-form").addEventListener("submit", (e)=>{ e.preventDefault(); send(input.value); });

/* ---------------- uploads ---------------- */
attachBtn.addEventListener("click", ()=>{
    const open = attachMenu.hidden;
    attachMenu.hidden = !open;
    attachBtn.setAttribute("aria-expanded", open ? "true" : "false");
});
attachMenu.querySelectorAll("button").forEach(b=>{
    b.addEventListener("click", ()=>{
        uploadKind = b.dataset.kind;
        attachMenu.hidden = true;
        attachBtn.setAttribute("aria-expanded", "false");
        fileInput.accept = uploadKind === "disease" ? ".csv,.xlsx,.xls" : ".csv,.xlsx,.xls,.parquet";
        fileInput.click();
    });
});
document.addEventListener("click", (e)=>{
    if(!attachMenu.hidden && !e.target.closest(".chat-attach")){ attachMenu.hidden = true; }
});
fileInput.addEventListener("change", async ()=>{
    const file = fileInput.files[0];
    fileInput.value = "";
    if(!file || busy) return;
    const form = new FormData();
    form.append("file", file);
    form.append("kind", uploadKind);
    setBusy(true);
    try{
        const r = await api("/assistant/upload", {method:"POST", body: form});
        if(!r.ok){
            const e = await r.json().catch(()=>({}));
            addBubble({role:"assistant", text: e.detail || `Upload failed (HTTP ${r.status}).`});
        }
    }catch(err){
        addBubble({role:"assistant", text:"Upload failed: could not reach the ClimAID server."});
    }
    poll();
});

/* ---------------- new conversation ---------------- */
document.getElementById("new-chat").addEventListener("click", async ()=>{
    if(busy) return;
    const r = await api("/assistant/reset", {method:"POST"});
    if(r.ok){ chat.innerHTML = ""; shown = 0; lastAssistantText = ""; poll(); }
});

setInterval(tickBusy, 1000);
poll();
input.focus();
