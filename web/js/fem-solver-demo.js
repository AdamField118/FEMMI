// Complete FEM Solver Demo
// Interactive demonstration of solving the lensing Poisson equation

(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('Code container not found!');
        return;
    }
    
    container.innerHTML = `
        <h3>Complete FEM Solver: ∇²ψ = 2κ</h3>
        <p>Place mass distributions (convergence κ), solve the Poisson equation using FEM, and visualize the resulting lensing potential and deflection field.</p>
        
        <div class="controls">
            <div style="display: flex; gap: 10px; justify-content: center; margin-bottom: 15px; flex-wrap: wrap;">
                <button id="solve-btn" style="width: 120px;">🧮 Solve FEM</button>
                <button id="clear-btn" style="width: 120px;">🗑️ Clear</button>
                <button id="preset-btn" style="width: 120px;">📦 Load Preset</button>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                <label style="display: flex; flex-direction: column;">
                    Mesh Resolution: <span id="res-val" style="font-weight: bold;">20</span>
                    <input type="range" id="res-slider" min="10" max="40" step="5" value="20" style="width: 100%;" />
                </label>
                <label style="display: flex; flex-direction: column;">
                    Mass Strength: <span id="mass-val" style="font-weight: bold;">1.0</span>
                    <input type="range" id="mass-slider" min="0.5" max="3.0" step="0.5" value="1.0" style="width: 100%;" />
                </label>
                <label style="display: flex; flex-direction: column;">
                    Visualization:
                    <select id="viz-select" style="width: 100%; padding: 5px;">
                        <option value="potential">Lensing Potential ψ</option>
                        <option value="deflection">Deflection Field α</option>
                        <option value="convergence">Convergence κ</option>
                    </select>
                </label>
            </div>
        </div>
        
        <canvas id="solver-canvas" width="700" height="700" style="border: 1px solid #ddd; width: 100%; max-width: 700px; display: block; margin: 20px auto; background: white; cursor: crosshair;"></canvas>
        
        <div id="solver-info" style="margin-top: 15px; padding: 15px; background: #e6f2ff; color: #000; border-radius: 5px; display: none;">
            <strong>Solver Statistics:</strong><br>
            • Mesh nodes: <span id="info-nodes">--</span><br>
            • Elements: <span id="info-elements">--</span><br>
            • Matrix size: <span id="info-matrix">--</span><br>
            • Nonzeros: <span id="info-nnz">--</span><br>
            • CG iterations: <span id="info-iter">--</span><br>
            • Residual: <span id="info-residual">--</span><br>
            • Max |ψ|: <span id="info-psi-max">--</span><br>
            • Max |α|: <span id="info-alpha-max">--</span>
        </div>
        
        <div style="margin-top: 15px; padding: 10px; background: #f0f0f0; color: #000; border-radius: 5px;">
            <strong>Instructions:</strong><br>
            1. Click on the canvas to place mass concentrations (convergence κ)<br>
            2. Click "Solve FEM" to assemble the system and solve Kψ = f<br>
            3. Switch visualization to see potential, deflection, or convergence<br>
            4. Try "Load Preset" for a galaxy cluster configuration<br>
            <br>
            <strong>Physics:</strong> The solver implements the weak form ∫∇ψ·∇v = ∫2κv using piecewise linear elements.
        </div>
    `;
    
    const canvas = container.querySelector('#solver-canvas');
    const ctx = canvas.getContext('2d');
    const solveBtn = container.querySelector('#solve-btn');
    const clearBtn = container.querySelector('#clear-btn');
    const presetBtn = container.querySelector('#preset-btn');
    const resSlider = container.querySelector('#res-slider');
    const massSlider = container.querySelector('#mass-slider');
    const vizSelect = container.querySelector('#viz-select');
    const resVal = container.querySelector('#res-val');
    const massVal = container.querySelector('#mass-val');
    const solverInfo = container.querySelector('#solver-info');
    
    let nodes = [];
    let elements = [];
    let masses = [];
    let psi = null;
    let alpha = null;
    let resolution = 20;
    let massStrength = 1.0;
    let vizMode = 'potential';
    
    resSlider.addEventListener('input', () => {
        resolution = parseInt(resSlider.value);
        resVal.textContent = resolution;
    });
    
    massSlider.addEventListener('input', () => {
        massStrength = parseFloat(massSlider.value);
        massVal.textContent = massStrength.toFixed(1);
    });
    
    vizSelect.addEventListener('change', () => {
        vizMode = vizSelect.value;
        draw();
    });
    
    solveBtn.addEventListener('click', () => {
        solve();
    });
    
    clearBtn.addEventListener('click', () => {
        masses = [];
        psi = null;
        alpha = null;
        solverInfo.style.display = 'none';
        draw();
    });
    
    presetBtn.addEventListener('click', () => {
        loadPreset();
    });
    
    canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        
        // Add mass
        masses.push({ x, y, m: massStrength });
        draw();
    });
    
    function loadPreset() {
        masses = [
            { x: 0.3, y: 0.4, m: 2.0 },
            { x: 0.7, y: 0.3, m: 1.5 },
            { x: 0.5, y: 0.7, m: 1.0 }
        ];
        draw();
    }
    
    function generateMesh(n) {
        nodes = [];
        elements = [];
        
        const step = 1 / n;
        
        // Create nodes
        for (let i = 0; i <= n; i++) {
            for (let j = 0; j <= n; j++) {
                nodes.push({ x: j * step, y: i * step });
            }
        }
        
        // Create elements
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const n0 = i * (n + 1) + j;
                const n1 = n0 + 1;
                const n2 = n0 + (n + 1);
                const n3 = n2 + 1;
                
                elements.push([n0, n1, n2]);
                elements.push([n1, n3, n2]);
            }
        }
    }
    
    function computeKappa(x, y) {
        let kappa = 0;
        
        masses.forEach(mass => {
            const dx = x - mass.x;
            const dy = y - mass.y;
            const r2 = dx * dx + dy * dy + 0.001; // Regularize
            
            // Simple 1/r model
            kappa += mass.m / Math.sqrt(r2);
        });
        
        return kappa;
    }
    
    function assembleSystem() {
        const n = nodes.length;
        
        // Initialize sparse storage (COO format)
        const rows = [];
        const cols = [];
        const vals = [];
        const f = Array(n).fill(0);
        
        // Assemble each element
        elements.forEach(elem => {
            const [n0, n1, n2] = elem;
            const p0 = nodes[n0];
            const p1 = nodes[n1];
            const p2 = nodes[n2];
            
            // Compute area
            const area = 0.5 * Math.abs(
                (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)
            );
            
            // Gradients
            const gradN = [
                { x: (p1.y - p2.y) / (2 * area), y: (p2.x - p1.x) / (2 * area) },
                { x: (p2.y - p0.y) / (2 * area), y: (p0.x - p2.x) / (2 * area) },
                { x: (p0.y - p1.y) / (2 * area), y: (p1.x - p0.x) / (2 * area) }
            ];
            
            // Element stiffness
            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) {
                    const K_ij = (gradN[i].x * gradN[j].x + gradN[i].y * gradN[j].y) * area;
                    
                    rows.push(elem[i]);
                    cols.push(elem[j]);
                    vals.push(K_ij);
                }
            }
            
            // Load vector (assume piecewise constant kappa at element centroid)
            const cx = (p0.x + p1.x + p2.x) / 3;
            const cy = (p0.y + p1.y + p2.y) / 3;
            const kappa = computeKappa(cx, cy);
            
            for (let i = 0; i < 3; i++) {
                f[elem[i]] += 2 * kappa * area / 3;
            }
        });
        
        // Convert COO to dense (for simplicity - real code would use sparse)
        const K = Array(n).fill(null).map(() => Array(n).fill(0));
        for (let i = 0; i < rows.length; i++) {
            K[rows[i]][cols[i]] += vals[i];
        }
        
        return { K, f, nnz: vals.length };
    }
    
    function applyBoundaryConditions(K, f) {
        const n = nodes.length;
        const gridSize = Math.sqrt(n);
        
        // Mark boundary nodes
        const isBoundary = Array(n).fill(false);
        for (let i = 0; i < n; i++) {
            const row = Math.floor(i / gridSize);
            const col = i % gridSize;
            
            if (row === 0 || row === gridSize - 1 || col === 0 || col === gridSize - 1) {
                isBoundary[i] = true;
            }
        }
        
        // Apply Dirichlet BC: psi = 0
        for (let i = 0; i < n; i++) {
            if (isBoundary[i]) {
                K[i] = Array(n).fill(0);
                K[i][i] = 1;
                f[i] = 0;
            }
        }
    }
    
    function conjugateGradient(K, f, maxIter = 1000, tol = 1e-6) {
        const n = K.length;
        let x = Array(n).fill(0);
        
        // r = f - K*x (initially r = f)
        let r = [...f];
        let p = [...r];
        let rsold = dot(r, r);
        
        let iter;
        for (iter = 0; iter < maxIter; iter++) {
            // Ap = K*p
            const Ap = matvec(K, p);
            
            // alpha = rsold / (p^T * Ap)
            const pAp = dot(p, Ap);
            if (Math.abs(pAp) < 1e-12) break;
            const alpha = rsold / pAp;
            
            // x = x + alpha*p
            for (let i = 0; i < n; i++) {
                x[i] += alpha * p[i];
            }
            
            // r = r - alpha*Ap
            for (let i = 0; i < n; i++) {
                r[i] -= alpha * Ap[i];
            }
            
            const rsnew = dot(r, r);
            
            // Check convergence
            if (Math.sqrt(rsnew) < tol) {
                break;
            }
            
            // p = r + (rsnew/rsold)*p
            const beta = rsnew / rsold;
            for (let i = 0; i < n; i++) {
                p[i] = r[i] + beta * p[i];
            }
            
            rsold = rsnew;
        }
        
        const residual = Math.sqrt(dot(r, r));
        return { x, iter, residual };
    }
    
    function dot(a, b) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    function matvec(K, x) {
        const n = K.length;
        const result = Array(n).fill(0);
        
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                result[i] += K[i][j] * x[j];
            }
        }
        
        return result;
    }
    
    function computeDeflection() {
        // Compute gradient of psi at each node (averaged from neighboring elements)
        const n = nodes.length;
        alpha = Array(n).fill(null).map(() => ({ x: 0, y: 0, count: 0 }));
        
        elements.forEach(elem => {
            const [n0, n1, n2] = elem;
            const p0 = nodes[n0];
            const p1 = nodes[n1];
            const p2 = nodes[n2];
            
            const area = 0.5 * Math.abs(
                (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)
            );
            
            // Gradients of shape functions
            const gradN = [
                { x: (p1.y - p2.y) / (2 * area), y: (p2.x - p1.x) / (2 * area) },
                { x: (p2.y - p0.y) / (2 * area), y: (p0.x - p2.x) / (2 * area) },
                { x: (p0.y - p1.y) / (2 * area), y: (p1.x - p0.x) / (2 * area) }
            ];
            
            // grad(psi) = sum_i psi_i * grad(N_i)
            const gradPsi = { x: 0, y: 0 };
            for (let i = 0; i < 3; i++) {
                gradPsi.x += psi[elem[i]] * gradN[i].x;
                gradPsi.y += psi[elem[i]] * gradN[i].y;
            }
            
            // Add to all nodes in element
            for (let i = 0; i < 3; i++) {
                alpha[elem[i]].x += gradPsi.x;
                alpha[elem[i]].y += gradPsi.y;
                alpha[elem[i]].count++;
            }
        });
        
        // Average
        for (let i = 0; i < n; i++) {
            if (alpha[i].count > 0) {
                alpha[i].x /= alpha[i].count;
                alpha[i].y /= alpha[i].count;
            }
        }
    }
    
    function solve() {
        generateMesh(resolution);
        
        const { K, f, nnz } = assembleSystem();
        applyBoundaryConditions(K, f);
        
        const result = conjugateGradient(K, f);
        psi = result.x;
        
        computeDeflection();
        
        // Update info display
        const maxPsi = Math.max(...psi.map(Math.abs));
        const maxAlpha = Math.max(...alpha.map(a => Math.sqrt(a.x * a.x + a.y * a.y)));
        
        container.querySelector('#info-nodes').textContent = nodes.length;
        container.querySelector('#info-elements').textContent = elements.length;
        container.querySelector('#info-matrix').textContent = `${nodes.length} × ${nodes.length}`;
        container.querySelector('#info-nnz').textContent = nnz;
        container.querySelector('#info-iter').textContent = result.iter;
        container.querySelector('#info-residual').textContent = result.residual.toExponential(2);
        container.querySelector('#info-psi-max').textContent = maxPsi.toFixed(4);
        container.querySelector('#info-alpha-max').textContent = maxAlpha.toFixed(4);
        
        solverInfo.style.display = 'block';
        draw();
    }
    
    function draw() {
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, width, height);
        
        if (vizMode === 'convergence' || masses.length === 0 || !psi) {
            // Draw convergence field
            const gridSize = 100;
            for (let i = 0; i < gridSize; i++) {
                for (let j = 0; j < gridSize; j++) {
                    const x = i / gridSize;
                    const y = j / gridSize;
                    const kappa = computeKappa(x, y);
                    
                    const intensity = Math.min(1, kappa / 2);
                    const color = Math.floor(255 * (1 - intensity));
                    
                    ctx.fillStyle = `rgb(255, ${color}, ${color})`;
                    ctx.fillRect(
                        i * width / gridSize,
                        j * height / gridSize,
                        width / gridSize,
                        height / gridSize
                    );
                }
            }
        }
        
        if (psi && vizMode === 'potential') {
            // Draw potential field
            const maxPsi = Math.max(...psi.map(Math.abs));
            const gridSize = Math.sqrt(nodes.length);
            
            for (let i = 0; i < nodes.length; i++) {
                const row = Math.floor(i / gridSize);
                const col = i % gridSize;
                
                if (row < gridSize - 1 && col < gridSize - 1) {
                    const val = psi[i] / (maxPsi + 0.01);
                    const intensity = Math.floor(255 * (0.5 + 0.5 * val));
                    
                    ctx.fillStyle = `rgb(${intensity}, ${intensity}, 255)`;
                    ctx.fillRect(
                        col * width / gridSize,
                        row * height / gridSize,
                        width / gridSize,
                        height / gridSize
                    );
                }
            }
        }
        
        if (alpha && vizMode === 'deflection') {
            // Draw deflection vectors
            const skip = Math.max(1, Math.floor(Math.sqrt(nodes.length) / 15));
            const scale = 2000;
            
            for (let i = 0; i < nodes.length; i += skip) {
                const node = nodes[i];
                const defl = alpha[i];
                
                const px = node.x * width;
                const py = node.y * height;
                
                const mag = Math.sqrt(defl.x * defl.x + defl.y * defl.y);
                const color = Math.floor(255 * Math.min(1, mag * 10));
                
                ctx.strokeStyle = `rgb(${color}, 0, ${255 - color})`;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(px, py);
                ctx.lineTo(px + defl.x * scale, py + defl.y * scale);
                ctx.stroke();
                
                // Arrowhead
                const angle = Math.atan2(defl.y, defl.x);
                const headLen = 5;
                ctx.beginPath();
                ctx.moveTo(px + defl.x * scale, py + defl.y * scale);
                ctx.lineTo(
                    px + defl.x * scale - headLen * Math.cos(angle - Math.PI / 6),
                    py + defl.y * scale - headLen * Math.sin(angle - Math.PI / 6)
                );
                ctx.moveTo(px + defl.x * scale, py + defl.y * scale);
                ctx.lineTo(
                    px + defl.x * scale - headLen * Math.cos(angle + Math.PI / 6),
                    py + defl.y * scale - headLen * Math.sin(angle + Math.PI / 6)
                );
                ctx.stroke();
            }
        }
        
        // Draw mass centers
        masses.forEach(mass => {
            const px = mass.x * width;
            const py = mass.y * height;
            
            ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
            ctx.beginPath();
            ctx.arc(px, py, 20, 0, 2 * Math.PI);
            ctx.fill();
            
            ctx.fillStyle = '#ff0000';
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, 2 * Math.PI);
            ctx.fill();
        });
        
        // Instructions if no masses
        if (masses.length === 0) {
            ctx.fillStyle = '#666';
            ctx.font = '18px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Click to place masses, then click Solve', width / 2, height / 2);
        }
    }
    
    draw();
    console.log('FEM solver demo loaded!');
})();
