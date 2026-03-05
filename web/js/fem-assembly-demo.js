// FEM Assembly Demo
// Visualizes how element stiffness matrices assemble into global system

(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('Code container not found!');
        return;
    }
    
    container.innerHTML = `
        <h3>FEM Assembly Process</h3>
        <p>Watch how individual 3×3 element stiffness matrices combine to build the global sparse system matrix K.</p>
        
        <div class="controls">
            <div style="display: flex; gap: 10px; justify-content: center; margin-bottom: 15px; flex-wrap: wrap;">
                <button id="step-btn" style="width: 140px;">▶ Step</button>
                <button id="auto-btn" style="width: 140px;">⏯ Auto</button>
                <button id="reset-btn" style="width: 140px;">🔄 Reset</button>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                <label style="display: flex; flex-direction: column;">
                    Animation Speed: <span id="speed-val" style="font-weight: bold;">1.0</span>
                    <input type="range" id="speed-slider" min="0.5" max="3.0" step="0.5" value="1.0" style="width: 100%;" />
                </label>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
            <div>
                <h4 style="margin-top: 0; text-align: center;">Mesh & Current Element</h4>
                <canvas id="mesh-canvas" width="400" height="400" style="border: 1px solid #ddd; width: 100%; background: white;"></canvas>
            </div>
            <div>
                <h4 style="margin-top: 0; text-align: center;">Global Stiffness Matrix K</h4>
                <canvas id="matrix-canvas" width="400" height="400" style="border: 1px solid #ddd; width: 100%; background: white;"></canvas>
            </div>
        </div>
        
        <div id="status-display" style="margin-top: 15px; padding: 15px; background: #e6f2ff; color: #000; border-radius: 5px;">
            <strong>Assembly Status:</strong> <span id="status-text">Ready</span><br>
            <strong>Current Element:</strong> <span id="elem-text">--</span> / <span id="total-elem">--</span><br>
            <strong>Global Nodes:</strong> <span id="global-nodes">--</span><br>
            <strong>Matrix Nonzeros:</strong> <span id="nonzeros">0</span>
        </div>
        
        <div style="margin-top: 15px;">
            <h4>Current Element Stiffness K<sup>e</sup>:</h4>
            <div id="element-matrix" style="font-family: monospace; background: #f5f5f5; padding: 10px; border-radius: 5px; color: #000;">
                <em>Select an element to see its stiffness matrix</em>
            </div>
        </div>
        
        <div style="margin-top: 15px; padding: 10px; background: #f0f0f0; color: #000; border-radius: 5px;">
            <strong>Assembly Formula:</strong> K[I,J] += K<sup>e</sup>[i,j] where I,J are global indices<br>
            • Each element contributes to 9 entries (3×3)<br>
            • Multiple elements contribute to shared nodes<br>
            • Result: sparse symmetric matrix
        </div>
    `;
    
    const meshCanvas = container.querySelector('#mesh-canvas');
    const matrixCanvas = container.querySelector('#matrix-canvas');
    const meshCtx = meshCanvas.getContext('2d');
    const matrixCtx = matrixCanvas.getContext('2d');
    const stepBtn = container.querySelector('#step-btn');
    const autoBtn = container.querySelector('#auto-btn');
    const resetBtn = container.querySelector('#reset-btn');
    const speedSlider = container.querySelector('#speed-slider');
    const speedVal = container.querySelector('#speed-val');
    const statusText = container.querySelector('#status-text');
    const elemText = container.querySelector('#elem-text');
    const totalElem = container.querySelector('#total-elem');
    const globalNodes = container.querySelector('#global-nodes');
    const nonzerosText = container.querySelector('#nonzeros');
    const elementMatrix = container.querySelector('#element-matrix');
    
    let nodes = [];
    let elements = [];
    let globalK = [];
    let currentElement = 0;
    let isAnimating = false;
    let animationSpeed = 1.0;
    let animationId = null;
    
    speedSlider.addEventListener('input', () => {
        animationSpeed = parseFloat(speedSlider.value);
        speedVal.textContent = animationSpeed.toFixed(1);
    });
    
    stepBtn.addEventListener('click', () => {
        if (currentElement < elements.length) {
            assembleElement(currentElement);
            currentElement++;
            draw();
        }
    });
    
    autoBtn.addEventListener('click', () => {
        if (!isAnimating) {
            isAnimating = true;
            autoBtn.textContent = '⏸ Pause';
            animate();
        } else {
            isAnimating = false;
            autoBtn.textContent = '⏯ Auto';
            if (animationId) clearTimeout(animationId);
        }
    });
    
    resetBtn.addEventListener('click', () => {
        reset();
    });
    
    function generateSimpleMesh() {
        nodes = [
            { x: 0.2, y: 0.2 },
            { x: 0.8, y: 0.2 },
            { x: 0.5, y: 0.8 },
            { x: 0.2, y: 0.8 },
            { x: 0.8, y: 0.8 }
        ];
        
        elements = [
            [0, 1, 2],  // Bottom triangle
            [0, 2, 3],  // Left triangle
            [1, 4, 2]   // Right triangle
        ];
    }
    
    function reset() {
        generateSimpleMesh();
        
        const n = nodes.length;
        globalK = Array(n).fill(null).map(() => Array(n).fill(0));
        currentElement = 0;
        isAnimating = false;
        if (animationId) clearTimeout(animationId);
        autoBtn.textContent = '⏯ Auto';
        
        statusText.textContent = 'Ready';
        elemText.textContent = '0';
        totalElem.textContent = elements.length;
        globalNodes.textContent = nodes.length;
        nonzerosText.textContent = '0';
        elementMatrix.innerHTML = '<em>Click Step to begin assembly</em>';
        
        draw();
    }
    
    function computeElementStiffness(elem) {
        const [n0, n1, n2] = elem;
        const p0 = nodes[n0];
        const p1 = nodes[n1];
        const p2 = nodes[n2];
        
        // Compute area
        const area = 0.5 * Math.abs((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y));
        
        // Gradients of shape functions
        const gradN = [
            { x: (p1.y - p2.y) / (2 * area), y: (p2.x - p1.x) / (2 * area) },
            { x: (p2.y - p0.y) / (2 * area), y: (p0.x - p2.x) / (2 * area) },
            { x: (p0.y - p1.y) / (2 * area), y: (p1.x - p0.x) / (2 * area) }
        ];
        
        // K^e[i][j] = grad_N_i · grad_N_j * area
        const Ke = Array(3).fill(null).map(() => Array(3).fill(0));
        
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                Ke[i][j] = (gradN[i].x * gradN[j].x + gradN[i].y * gradN[j].y) * area;
            }
        }
        
        return Ke;
    }
    
    function assembleElement(elemIdx) {
        const elem = elements[elemIdx];
        const Ke = computeElementStiffness(elem);
        
        // Add to global matrix
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                const I = elem[i];
                const J = elem[j];
                globalK[I][J] += Ke[i][j];
            }
        }
        
        // Update display
        statusText.textContent = 'Assembling...';
        elemText.textContent = (elemIdx + 1).toString();
        
        // Count nonzeros
        let nnz = 0;
        for (let i = 0; i < globalK.length; i++) {
            for (let j = 0; j < globalK[i].length; j++) {
                if (Math.abs(globalK[i][j]) > 1e-10) nnz++;
            }
        }
        nonzerosText.textContent = nnz;
        
        // Display element matrix
        let matrixHTML = `<strong>Element ${elemIdx + 1}</strong> (nodes: ${elem.join(', ')})<br><br>`;
        matrixHTML += '<table style="border-collapse: collapse; margin: 0 auto;">';
        for (let i = 0; i < 3; i++) {
            matrixHTML += '<tr>';
            for (let j = 0; j < 3; j++) {
                matrixHTML += `<td style="padding: 5px; border: 1px solid #ccc; text-align: center;">${Ke[i][j].toFixed(4)}</td>`;
            }
            matrixHTML += '</tr>';
        }
        matrixHTML += '</table><br>';
        matrixHTML += `<strong>Adds to global positions:</strong><br>`;
        matrixHTML += `K[${elem[0]},${elem[0]}], K[${elem[0]},${elem[1]}], K[${elem[0]},${elem[2]}]<br>`;
        matrixHTML += `K[${elem[1]},${elem[0]}], K[${elem[1]},${elem[1]}], K[${elem[1]},${elem[2]}]<br>`;
        matrixHTML += `K[${elem[2]},${elem[0]}], K[${elem[2]},${elem[1]}], K[${elem[2]},${elem[2]}]`;
        
        elementMatrix.innerHTML = matrixHTML;
        
        if (elemIdx === elements.length - 1) {
            statusText.textContent = 'Assembly complete!';
            isAnimating = false;
            autoBtn.textContent = '⏯ Auto';
        }
    }
    
    function animate() {
        if (!isAnimating || currentElement >= elements.length) {
            isAnimating = false;
            autoBtn.textContent = '⏯ Auto';
            return;
        }
        
        assembleElement(currentElement);
        currentElement++;
        draw();
        
        animationId = setTimeout(() => animate(), 1000 / animationSpeed);
    }
    
    function draw() {
        drawMesh();
        drawMatrix();
    }
    
    function drawMesh() {
        const width = meshCanvas.width;
        const height = meshCanvas.height;
        
        meshCtx.fillStyle = 'white';
        meshCtx.fillRect(0, 0, width, height);
        
        // Draw all elements
        elements.forEach((elem, idx) => {
            const [n0, n1, n2] = elem;
            const p0 = nodes[n0];
            const p1 = nodes[n1];
            const p2 = nodes[n2];
            
            // Highlight current/completed elements
            if (idx < currentElement) {
                meshCtx.fillStyle = idx === currentElement - 1 ? 'rgba(255, 0, 0, 0.2)' : 'rgba(0, 200, 0, 0.1)';
                meshCtx.beginPath();
                meshCtx.moveTo(p0.x * width, p0.y * height);
                meshCtx.lineTo(p1.x * width, p1.y * height);
                meshCtx.lineTo(p2.x * width, p2.y * height);
                meshCtx.closePath();
                meshCtx.fill();
            }
            
            // Draw edges
            meshCtx.strokeStyle = idx === currentElement - 1 ? '#ff0000' : '#333';
            meshCtx.lineWidth = idx === currentElement - 1 ? 3 : 1;
            meshCtx.beginPath();
            meshCtx.moveTo(p0.x * width, p0.y * height);
            meshCtx.lineTo(p1.x * width, p1.y * height);
            meshCtx.lineTo(p2.x * width, p2.y * height);
            meshCtx.closePath();
            meshCtx.stroke();
        });
        
        // Draw nodes
        nodes.forEach((node, i) => {
            const px = node.x * width;
            const py = node.y * height;
            
            meshCtx.fillStyle = '#0066cc';
            meshCtx.beginPath();
            meshCtx.arc(px, py, 5, 0, 2 * Math.PI);
            meshCtx.fill();
            
            // Node label
            meshCtx.fillStyle = '#000';
            meshCtx.font = '12px Arial';
            meshCtx.fillText(i.toString(), px + 8, py - 8);
        });
        
        // Element labels
        elements.forEach((elem, idx) => {
            const [n0, n1, n2] = elem;
            const p0 = nodes[n0];
            const p1 = nodes[n1];
            const p2 = nodes[n2];
            
            const cx = (p0.x + p1.x + p2.x) / 3 * width;
            const cy = (p0.y + p1.y + p2.y) / 3 * height;
            
            meshCtx.fillStyle = idx < currentElement ? '#ff0000' : '#999';
            meshCtx.font = 'bold 14px Arial';
            meshCtx.textAlign = 'center';
            meshCtx.fillText(`e${idx}`, cx, cy);
        });
    }
    
    function drawMatrix() {
        const width = matrixCanvas.width;
        const height = matrixCanvas.height;
        
        matrixCtx.fillStyle = 'white';
        matrixCtx.fillRect(0, 0, width, height);
        
        const n = nodes.length;
        const cellSize = Math.min(width, height) / (n + 2);
        const offsetX = (width - n * cellSize) / 2;
        const offsetY = (height - n * cellSize) / 2;
        
        // Find max value for color scaling
        let maxVal = 0;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                maxVal = Math.max(maxVal, Math.abs(globalK[i][j]));
            }
        }
        
        // Draw matrix cells
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const val = globalK[i][j];
                const intensity = maxVal > 0 ? Math.abs(val) / maxVal : 0;
                
                // Color: white (0) to blue (max)
                const color = Math.floor(255 * (1 - intensity));
                matrixCtx.fillStyle = `rgb(${color}, ${color}, 255)`;
                matrixCtx.fillRect(
                    offsetX + j * cellSize,
                    offsetY + i * cellSize,
                    cellSize,
                    cellSize
                );
                
                // Border
                matrixCtx.strokeStyle = '#ccc';
                matrixCtx.lineWidth = 0.5;
                matrixCtx.strokeRect(
                    offsetX + j * cellSize,
                    offsetY + i * cellSize,
                    cellSize,
                    cellSize
                );
            }
        }
        
        // Draw axis labels
        matrixCtx.fillStyle = '#000';
        matrixCtx.font = '12px Arial';
        matrixCtx.textAlign = 'center';
        
        for (let i = 0; i < n; i++) {
            // Row labels
            matrixCtx.fillText(
                i.toString(),
                offsetX - 15,
                offsetY + i * cellSize + cellSize / 2 + 4
            );
            
            // Column labels
            matrixCtx.fillText(
                i.toString(),
                offsetX + i * cellSize + cellSize / 2,
                offsetY - 10
            );
        }
        
        // Matrix label
        matrixCtx.font = 'bold 14px Arial';
        matrixCtx.fillText('K', offsetX - 30, offsetY + n * cellSize / 2);
        
        // Legend
        matrixCtx.font = '10px Arial';
        matrixCtx.textAlign = 'left';
        matrixCtx.fillText('0', 10, height - 10);
        matrixCtx.textAlign = 'right';
        matrixCtx.fillText(`max: ${maxVal.toFixed(4)}`, width - 10, height - 10);
        
        // Draw color bar
        const barWidth = 100;
        const barHeight = 10;
        for (let i = 0; i < barWidth; i++) {
            const intensity = i / barWidth;
            const color = Math.floor(255 * (1 - intensity));
            matrixCtx.fillStyle = `rgb(${color}, ${color}, 255)`;
            matrixCtx.fillRect(10 + i, height - 30, 1, barHeight);
        }
        matrixCtx.strokeStyle = '#000';
        matrixCtx.strokeRect(10, height - 30, barWidth, barHeight);
    }
    
    reset();
    console.log('FEM assembly demo loaded!');
})();
