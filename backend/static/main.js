// main.js
// main.js
async function fetchPatches() {
  const hostInput = document.getElementById("hostname");
  const host = hostInput.value.trim(); // || "DESKTOP-M9KGOJ7"; // default if empty
  const out = document.getElementById("patchList");
  out.innerHTML = "<p>Loading...</p>";

  try {
    const res = await fetch(`/api/patches/${encodeURIComponent(host)}`);
    if (!res.ok) {
      out.innerHTML = "<p>No data for that host. Run scan on host first.</p>";
      return;
    }
    const data = await res.json();
    if (!data.patches || data.patches.length === 0) {
      out.innerHTML = "<p>No patches found.</p>";
      return;
    }

    let html = "<table><tr><th>KB</th><th>Title</th><th>Severity</th><th>Priority</th><th>Action</th></tr>";
    for (const p of data.patches) {
      html += `<tr>
        <td>${p.kb}</td>
        <td>${p.title}</td>
        <td>${p.severity}</td>
        <td>${(p.priority_score || 0).toFixed(3)}</td>
        <td><button style="background: #27C2F5;"  onclick="triggerInstall('${data.host}','${p.kb}')">Get Install Cmd</button></td>
      </tr>`;
    }
    html += "</table>";
    out.innerHTML = html;
  } catch (err) {
    out.innerHTML = "<p>Error fetching patches: " + err + "</p>";
  }
}

async function triggerInstall(host, kb) {
  try {
    const res = await fetch("/api/trigger_install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hostname: host, kb: kb })
    });
    const data = await res.json();
    alert("Instruction:\n" + data.instruction);
  } catch (err) {
    alert("Failed to get install command: " + err);
  }
}

// Auto-fill hostname input with local hostname from backend
async function initHostname() {
  try {
    const res = await fetch("/api/get_hostname");
    if (res.ok) {
      const data = await res.json();
      document.getElementById("hostname").value = data.hostname;
    }
  } catch (err) {
    console.log("Failed to get hostname:", err);
  }
}

document.getElementById("btnFetch").addEventListener("click", fetchPatches);
initHostname();




document.addEventListener("DOMContentLoaded", () => {
  const btnDelete = document.getElementById("btndelete");
  const hostnameInput = document.getElementById("hostname");
  const patchList = document.getElementById("patchList");

 

  btnDelete.addEventListener("click", async () => {
    const hostname = hostnameInput.value.trim();

    if (!hostname) return alert("Enter a hostname first");
    if (!confirm(`Delete all patch entries for ${hostname}?`)) return;

    try {
      const res = await fetch(`/api/patches/${hostname}`, { method: "DELETE" });
      const data = await res.json();
      //alert(`Deleted patches for ${data.host}`);
      patchList.innerHTML = "<p>No patches found.</p>";
    } catch (err) {
      console.error(err);
      alert("Failed to delete patches");
    }
  });
});




document.addEventListener("DOMContentLoaded", () => {
  const btnScan = document.getElementById("btnScan");
  const scanOutput = document.getElementById("scanOutput");

  btnScan.addEventListener("click", async () => {
    scanOutput.textContent = "Running scan...";
    try {
      const res = await fetch("/api/run_scan", { method: "POST" });
      const data = await res.json();
      scanOutput.textContent = `STDOUT:\n${data.stdout}\n\nSTDERR:\n${data.stderr}`;
    } catch (err) {
      console.error(err);
      scanOutput.textContent = "Failed to run scan";
    }
  });
});


// main.js

async function fetchPatches22() {
    const host = document.getElementById("hostname").value.trim();
    const out = document.getElementById("patchList");
    out.innerHTML = "<p>Loading...</p>";

    try {
        const res = await fetch(`/api/patches/${encodeURIComponent(host)}`);
        if (!res.ok) {
            out.innerHTML = "<p>No data for that host. Run scan on host first.</p>";
            return;
        }

        const data = await res.json();

        if (!data.patches || data.patches.length === 0) {
            out.innerHTML = "<p>No patches found.</p>";
            return;
        }

        // Build table
        let html = `<table border="1" cellpadding="5">
                        <tr>
                            <th>KB</th>
                            <th>Title</th>
                            <th>Severity</th>
                            <th>Priority</th>
                            <th>Action</th>
                        </tr>`;

        for (const p of data.patches) {
            html += `<tr>
                        <td>${p.KB}</td>
                        <td>${p.Title}</td>
                        <td>${p.Severity}</td>
                        <td>${(p.Priority || 0).toFixed(3)}</td>
                        <td>
                            <button style="background: #27C2F5;"  onclick="triggerInstall('${data.hostname}','${p.KB}')">
                                Get Install Cmd
                            </button>
                        </td>
                     </tr>`;
        }

        html += "</table>";
        out.innerHTML = html;

    } catch (err) {
        out.innerHTML = "<p>Error fetching patches: " + err + "</p>";
    }
}

async function triggerInstall(host, kb) {
    try {
        const res = await fetch("/api/trigger_install", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ hostname: host, kb: kb })
        });

        if (!res.ok) {
            alert("Error triggering install: " + res.statusText);
            return;
        }

        const data = await res.json();
        if (!data.instruction) {
            alert("No install command returned from server.");
            return;
        }

        alert("Instruction: " + data.instruction);

    } catch (err) {
        alert("Error: " + err);
    }
}

// Attach event listener to fetch button
document.getElementById("btnFetch").addEventListener("click", fetchPatches);





// Auto-fill the hostname input with system hostname from backend
async function initHostname2() {
  try {
    const res = await fetch("/api/get_hostname");
    if (res.ok) {
      const data = await res.json();
      document.getElementById("hostname").value = data.hostname;
    }
  } catch (err) {
    console.log("Failed to get hostname:", err);
  }
}
initHostname();





async function fetchPatches2() {
  const host = document.getElementById("hostname").value.trim();
  const out = document.getElementById("patchList");
  out.innerHTML = "<p>Loading...</p>";
  try {
    const res = await fetch(`/api/patches/${encodeURIComponent(host)}`);
    if (!res.ok) {
      out.innerHTML = "<p>No data for that host. Run scan on host first.</p>";
      return;
    }
    const data = await res.json();
    if (!data.patches || data.patches.length===0) {
      out.innerHTML = "<p>No patches found.</p>";
      return;
    }
    let html = "<table><tr><th>KB</th><th>Title</th><th>Severity</th><th>Priority</th><th>Action</th></tr>";
    for (const p of data.patches) {
      html += `<tr>
        <td>${p.kb}</td>
        <td>${p.title}</td>
        <td>${p.severity}</td>
        <td>${(p.priority_score||0).toFixed(3)}</td>
        <td><button style="background: #27C2F5;"  onclick="triggerInstall('${data.host}','${p.kb}')">Get Install Cmd</button></td>
      </tr>`;
    }
    html += "</table>";
    out.innerHTML = html;
  } catch (err) {
    out.innerHTML = "<p>Error fetching patches: "+err+"</p>";
  }
}

async function triggerInstall(host, kb) {
  const res = await fetch("/api/trigger_install", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({hostname: host, kb: kb})
  });
  const data = await res.json();
  alert("Instruction: " + data.instruction);
}

document.getElementById("btnFetch").addEventListener("click", fetchPatches);
