// FEM Mesh and Basis Functions Demo
// Interactive visualization of 2D triangular mesh and piecewise linear basis functions

(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('Code container not found!');
        return;
    }
    
    container.innerHTML = `
        <h3>FEM Mesh and Basis Functions</h3>
        <p>Click on the mesh to see how basis functions (hat functions) work. Each node has a basis function that is 1 at that node and 0 at all others.</p>
        
        <div class="controls">
            <div style="display: flex; gap: 10px; justify-content: center; margin-bottom: 15px; flex-wrap: wrap;">
                <button id="generate-mesh-btn" style="width: 150px;">🔄 New Mesh</button>
                <button id="refine-btn" style="width: 150px;">➕ Refine</button>
                <button id="clear-selection-btn" style="width: 150px;">✖️ Clear Selection</button>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                <label style="display: flex; flex-direction: column;">
                    Mesh Resolution: <span id="resolution-val" style="font-weight: bold;">5</span>
                    <input type="range" id="resolution-slider" min="3" max="10" step="1" value="5" style="width: 100%;" />
                </label>
                <label style="display: flex; flex-direction: column;">
                    Visualization: 
                    <select id="viz-mode" style="width: 100%; padding: 5px;">
                        <option value="mesh">Mesh Only</option>
                        <option value="basis">Basis Function</option>
                        <option value="gradient">Gradient Field</option>
                    </select>
                </label>
            </div>
        </div>
        
        <canvas id="mesh-canvas" width="600" height="600" style="border: 1px solid #ddd; width: 100%; max-width: 600px; display: block; margin: 20px auto; background: white; cursor: crosshair;"></canvas>
        
        <div id="info-display" style="margin-top: 15px; padding: 15px; background: #e6f2ff; color: #000; border-radius: 5px; display: none;">
            <strong>Selected Node:</strong> <span id="node-info"></span><br>
            <strong>Basis Function φ<sub id="node-index"></sub>:</strong><br>
            • Value at node: <span id="phi-value">1.0</span><br>
            • Support: <span id="phi-support">--</span> elements<br>
            • Gradient: ∇φ = <span id="phi-gradient">--</span> (piecewise constant)
        </div>
        
        <div style="margin-top: 15px; padding: 10px; background: #f0f0f0; color: #000; border-radius: 5px;">
            <strong>How it works:</strong><br>
            • Each triangle has 3 nodes with 3 basis functions<br>
            • Basis functions are linear on each triangle (barycentric coordinates)<br>
            • φᵢ(θⱼ) = δᵢⱼ (Kronecker delta)<br>
            • Solution: ψ(θ) = Σᵢ ψᵢ φᵢ(θ)<br>
            <strong>Click a node</strong> to see its basis function visualized in color!
        </div>
    `;
    
    const canvas = container.querySelector('#mesh-canvas');
    const ctx = canvas.getContext('2d');
    const generateBtn = container.querySelector('#generate-mesh-btn');
    const refineBtn = container.querySelector('#refine-btn');
    const clearBtn = container.querySelector('#clear-selection-btn');
    const resolutionSlider = container.querySelector('#resolution-slider');
    const vizModeSelect = container.querySelector('#viz-mode');
    const resolutionVal = container.querySelector('#resolution-val');
    const infoDisplay = container.querySelector('#info-display');
    const nodeInfo = container.querySelector('#node-info');
    const nodeIndex = container.querySelector('#node-index');
    const phiValue = container.querySelector('#phi-value');
    const phiSupport = container.querySelector('#phi-support');
    const phiGradient = container.querySelector('#phi-gradient');
    
    let nodes = [];
    let elements = [];
    let resolution = 5;
    let selectedNode = null;
    let vizMode = 'mesh';
    
    resolutionSlider.addEventListener('input', () => {
        resolution = parseInt(resolutionSlider.value);
        resolutionVal.textContent = resolution;
    });
    
    generateBtn.addEventListener('click', () => {
        generateMesh(resolution);
        selectedNode = null;
        infoDisplay.style.display = 'none';
        draw();
    });
    
    refineBtn.addEventListener('click', () => {
        refineMesh();
        draw();
    });
    
    clearBtn.addEventListener('click', () => {
        selectedNode = null;
        infoDisplay.style.display = 'none';
        draw();
    });
    
    vizModeSelect.addEventListener('change', () => {
        vizMode = vizModeSelect.value;
        draw();
    });
    
    canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        
        // Find nearest node
        let minDist = Infinity;
        let nearestNode = null;
        
        nodes.forEach((node, i) => {
            const dx = node.x - x;
            const dy = node.y - y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist < minDist && dist < 0.05) { // Within 5% of canvas
                minDist = dist;
                nearestNode = i;
            }
        });
        
        if (nearestNode !== null) {
            selectedNode = nearestNode;
            updateNodeInfo();
            draw();
        }
    });
    
    function generateMesh(n) {
        nodes = [];
        elements = [];
        
        // Generate regular triangular mesh
        const step = 1 / n;
        
        // Create nodes
        for (let i = 0; i <= n; i++) {
            for (let j = 0; j <= n; j++) {
                nodes.push({
                    x: j * step,
                    y: i * step,
                    id: nodes.length
                });
            }
        }
        
        // Create triangular elements
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const n0 = i * (n + 1) + j;
                const n1 = n0 + 1;
                const n2 = n0 + (n + 1);
                const n3 = n2 + 1;
                
                // Lower triangle
                elements.push([n0, n1, n2]);
                // Upper triangle
                elements.push([n1, n3, n2]);
            }
        }
    }
    
    function refineMesh() {
        // Simple uniform refinement: split each triangle into 4
        const newNodes = [...nodes];
        const newElements = [];
        const edgeMap = new Map();
        
        function getEdgeMidpoint(n1, n2) {
            const key = n1 < n2 ? `${n1}-${n2}` : `${n2}-${n1}`;
            
            if (edgeMap.has(key)) {
                return edgeMap.get(key);
            }
            
            const node1 = nodes[n1];
            const node2 = nodes[n2];
            const mid = {
                x: (node1.x + node2.x) / 2,
                y: (node1.y + node2.y) / 2,
                id: newNodes.length
            };
            
            newNodes.push(mid);
            edgeMap.set(key, mid.id);
            return mid.id;
        }
        
        elements.forEach(elem => {
            const [n0, n1, n2] = elem;
            
            // Get midpoint nodes
            const m01 = getEdgeMidpoint(n0, n1);
            const m12 = getEdgeMidpoint(n1, n2);
            const m20 = getEdgeMidpoint(n2, n0);
            
            // Create 4 new triangles
            newElements.push([n0, m01, m20]);
            newElements.push([m01, n1, m12]);
            newElements.push([m20, m12, n2]);
            newElements.push([m01, m12, m20]);
        });
        
        nodes = newNodes;
        elements = newElements;
    }
    
    function computeBasisFunction(nodeIdx, x, y) {
        // Find which element contains (x, y) and compute basis function value
        
        for (let elem of elements) {
            const [n0, n1, n2] = elem;
            const p0 = nodes[n0];
            const p1 = nodes[n1];
            const p2 = nodes[n2];
            
            // Check if point is in this triangle using barycentric coordinates
            const denom = (p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y);
            const lambda0 = ((p1.y - p2.y) * (x - p2.x) + (p2.x - p1.x) * (y - p2.y)) / denom;
            const lambda1 = ((p2.y - p0.y) * (x - p2.x) + (p0.x - p2.x) * (y - p2.y)) / denom;
            const lambda2 = 1 - lambda0 - lambda1;
            
            if (lambda0 >= -0.001 && lambda1 >= -0.001 && lambda2 >= -0.001) {
                // Point is in this triangle
                if (n0 === nodeIdx) return lambda0;
                if (n1 === nodeIdx) return lambda1;
                if (n2 === nodeIdx) return lambda2;
                return 0; // Node not in this element
            }
        }
        
        return 0; // Outside all elements
    }
    
    function computeGradient(elem, nodeLocalIdx) {
        const [n0, n1, n2] = elem;
        const p0 = nodes[n0];
        const p1 = nodes[n1];
        const p2 = nodes[n2];
        
        // Compute area
        const area = 0.5 * Math.abs((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y));
        
        // Gradients of shape functions
        const grads = [
            { x: (p1.y - p2.y) / (2 * area), y: (p2.x - p1.x) / (2 * area) },
            { x: (p2.y - p0.y) / (2 * area), y: (p0.x - p2.x) / (2 * area) },
            { x: (p0.y - p1.y) / (2 * area), y: (p1.x - p0.x) / (2 * area) }
        ];
        
        return grads[nodeLocalIdx];
    }
    
    function updateNodeInfo() {
        if (selectedNode === null) return;
        
        const node = nodes[selectedNode];
        nodeInfo.textContent = `(${node.x.toFixed(3)}, ${node.y.toFixed(3)})`;
        nodeIndex.textContent = selectedNode;
        
        // Count support (elements containing this node)
        let support = 0;
        elements.forEach(elem => {
            if (elem.includes(selectedNode)) support++;
        });
        
        phiSupport.textContent = support;
        
        // Compute average gradient
        if (vizMode === 'gradient') {
            let avgGradX = 0;
            let avgGradY = 0;
            let count = 0;
            
            elements.forEach(elem => {
                const localIdx = elem.indexOf(selectedNode);
                if (localIdx !== -1) {
                    const grad = computeGradient(elem, localIdx);
                    avgGradX += grad.x;
                    avgGradY += grad.y;
                    count++;
                }
            });
            
            if (count > 0) {
                avgGradX /= count;
                avgGradY /= count;
                phiGradient.textContent = `(${avgGradX.toFixed(3)}, ${avgGradY.toFixed(3)})`;
            }
        }
        
        infoDisplay.style.display = 'block';
    }
    
    function draw() {
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, width, height);
        
        // Draw basis function as colored field
        if (vizMode === 'basis' && selectedNode !== null) {
            const gridSize = 100;
            const cellWidth = width / gridSize;
            const cellHeight = height / gridSize;
            
            for (let i = 0; i < gridSize; i++) {
                for (let j = 0; j < gridSize; j++) {
                    const x = i / gridSize;
                    const y = j / gridSize;
                    
                    const phi = computeBasisFunction(selectedNode, x, y);
                    
                    // Color: blue (0) to red (1)
                    const intensity = Math.floor(255 * phi);
                    ctx.fillStyle = `rgb(${intensity}, 0, ${255 - intensity})`;
                    ctx.fillRect(i * cellWidth, j * cellHeight, cellWidth, cellHeight);
                }
            }
        }
        
        // Draw gradient field
        if (vizMode === 'gradient' && selectedNode !== null) {
            elements.forEach(elem => {
                const localIdx = elem.indexOf(selectedNode);
                if (localIdx !== -1) {
                    const [n0, n1, n2] = elem;
                    const p0 = nodes[n0];
                    const p1 = nodes[n1];
                    const p2 = nodes[n2];
                    
                    // Centroid
                    const cx = (p0.x + p1.x + p2.x) / 3 * width;
                    const cy = (p0.y + p1.y + p2.y) / 3 * height;
                    
                    // Gradient
                    const grad = computeGradient(elem, localIdx);
                    const scale = 50;
                    
                    ctx.strokeStyle = '#ff0000';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.lineTo(cx + grad.x * scale, cy + grad.y * scale);
                    ctx.stroke();
                    
                    // Arrowhead
                    const angle = Math.atan2(grad.y, grad.x);
                    const headLen = 5;
                    ctx.beginPath();
                    ctx.moveTo(cx + grad.x * scale, cy + grad.y * scale);
                    ctx.lineTo(
                        cx + grad.x * scale - headLen * Math.cos(angle - Math.PI / 6),
                        cy + grad.y * scale - headLen * Math.sin(angle - Math.PI / 6)
                    );
                    ctx.moveTo(cx + grad.x * scale, cy + grad.y * scale);
                    ctx.lineTo(
                        cx + grad.x * scale - headLen * Math.cos(angle + Math.PI / 6),
                        cy + grad.y * scale - headLen * Math.sin(angle + Math.PI / 6)
                    );
                    ctx.stroke();
                }
            });
        }
        
        // Draw mesh edges
        ctx.strokeStyle = vizMode === 'mesh' ? '#333' : '#999';
        ctx.lineWidth = vizMode === 'mesh' ? 1.5 : 0.5;
        
        elements.forEach(elem => {
            const [n0, n1, n2] = elem;
            const p0 = nodes[n0];
            const p1 = nodes[n1];
            const p2 = nodes[n2];
            
            ctx.beginPath();
            ctx.moveTo(p0.x * width, p0.y * height);
            ctx.lineTo(p1.x * width, p1.y * height);
            ctx.lineTo(p2.x * width, p2.y * height);
            ctx.closePath();
            ctx.stroke();
        });
        
        // Draw nodes
        nodes.forEach((node, i) => {
            const px = node.x * width;
            const py = node.y * height;
            
            if (i === selectedNode) {
                ctx.fillStyle = '#ff0000';
                ctx.beginPath();
                ctx.arc(px, py, 6, 0, 2 * Math.PI);
                ctx.fill();
                
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.stroke();
            } else {
                ctx.fillStyle = '#0066cc';
                ctx.beginPath();
                ctx.arc(px, py, 3, 0, 2 * Math.PI);
                ctx.fill();
            }
        });
        
        // Draw legend
        if (vizMode === 'basis' && selectedNode !== null) {
            ctx.fillStyle = '#000';
            ctx.font = '12px Arial';
            ctx.fillText('φ = 0 (blue) → φ = 1 (red)', 10, height - 10);
        }
    }
    
    generateMesh(resolution);
    draw();
    console.log('FEM mesh demo loaded!');
})();
