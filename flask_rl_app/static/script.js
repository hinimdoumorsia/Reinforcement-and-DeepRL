class RLGridWorld {
    constructor() {
        this.gridCanvas = document.getElementById("gridCanvas");
        this.gridCtx = this.gridCanvas.getContext("2d");
        this.errorCanvas = document.getElementById("errorCanvas");
        this.errorCtx = this.errorCanvas.getContext("2d");
        
        this.simulationRunning = false;
        this.simulationSpeed = 200;
        this.currentData = null;
        
        this.initializeEventListeners();
        this.startDataPolling();
        this.setupCanvasStyles();
    }

    setupCanvasStyles() {
        // Style pour le thème sombre
        this.gridCanvas.style.background = '#0f172a';
        this.errorCanvas.style.background = '#0f172a';
    }

    // AJOUT DE LA MÉTHODE DE DÉBOGAGE ICI
    logTrainingInfo() {
        if (!this.currentData) return;
        
        console.log('Training Info:', {
            agent: this.currentData.agent_name,
            episode: this.currentData.current_episode,
            totalEpisodes: this.currentData.total_episodes,
            errors: this.currentData.errors?.length || 0,
            trainingComplete: this.currentData.training_complete
        });
    }

    initializeEventListeners() {
        // Form submission
        document.getElementById('simulationForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.startTraining();
        });

        // Speed control
        document.getElementById('speedControl').addEventListener('input', (e) => {
            this.updateSpeed(e.target.value);
        });

        // Agent selection
        document.getElementById('agentSelect').addEventListener('change', (e) => {
            this.updateAgentParameters(e.target.value);
        });
    }

    async startTraining() {
        const formData = new FormData(document.getElementById('simulationForm'));
        const data = Object.fromEntries(formData);
        
        // Convert numbers
        data.grid_size = parseInt(data.grid_size);
        data.episodes = parseInt(data.episodes);
        data.num_goals = parseInt(data.num_goals);
        data.num_obstacles = parseInt(data.num_obstacles);
        data.alpha = parseFloat(data.alpha);
        data.gamma = parseFloat(data.gamma);
        data.epsilon = parseFloat(data.epsilon);

        try {
            const response = await fetch('/start_training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            if (response.ok) {
                this.updateTrainingStatus('Training started...', 'training');
                document.getElementById('startTraining').disabled = true;
            }
        } catch (error) {
            console.error('Error starting training:', error);
            this.updateTrainingStatus('Error starting training', 'error');
        }
    }

    updateTrainingStatus(message, type = 'ready') {
        const statusElement = document.getElementById('trainingStatus');
        statusElement.textContent = message;
        
        statusElement.className = 'status-badge';
        if (type === 'training') {
            statusElement.classList.add('status-training', 'loading');
        } else if (type === 'complete') {
            statusElement.classList.add('status-complete');
        } else if (type === 'error') {
            statusElement.classList.add('status-error');
        } else {
            statusElement.classList.add('status-ready');
        }
    }

    updateAgentParameters(agentName) {
        console.log('Selected agent:', agentName);
    }

    updateSpeed(speed) {
        this.simulationSpeed = 500 - (speed * 45);
    }

    startDataPolling() {
        setInterval(() => {
            this.fetchData();
        }, 500);
    }

    async fetchData() {
        try {
            const response = await fetch('/data');
            this.currentData = await response.json();
            this.updateDisplay();
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    }

    // MODIFICATION DE updateDisplay() POUR AJOUTER LE DÉBOGAGE
    updateDisplay() {
        if (!this.currentData) return;

        // AJOUT: Log des informations de training
        this.logTrainingInfo();

        // Update status
        if (this.currentData.training_complete) {
            this.updateTrainingStatus('Training Complete', 'complete');
            document.getElementById('startTraining').disabled = false;
        } else if (this.currentData.current_episode > 0) {
            this.updateTrainingStatus(`Training: ${this.currentData.current_episode}/${this.currentData.total_episodes}`, 'training');
        } else {
            this.updateTrainingStatus('Ready to train', 'ready');
        }

        // Update episode info
        document.getElementById('episodeInfo').textContent = 
            `${this.currentData.current_episode}/${this.currentData.total_episodes}`;

        // Draw visualizations
        this.drawGrid();
        this.drawError();
    }

    drawGrid() {
        const { size, positions, goals, obstacles } = this.currentData;
        const cellSize = this.gridCanvas.width / size;
        
        // Clear avec couleur de fond sombre
        this.gridCtx.fillStyle = '#0f172a';
        this.gridCtx.fillRect(0, 0, this.gridCanvas.width, this.gridCanvas.height);

        // Draw grid lines
        this.gridCtx.strokeStyle = '#334155';
        this.gridCtx.lineWidth = 1;
        for (let y = 0; y <= size; y++) {
            for (let x = 0; x <= size; x++) {
                this.gridCtx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize);
            }
        }

        // Draw obstacles
        obstacles.forEach(obs => {
            this.gridCtx.fillStyle = '#475569';
            this.gridCtx.fillRect(obs[0] * cellSize, obs[1] * cellSize, cellSize, cellSize);
        });

        // Draw goals
        goals.forEach(goal => {
            this.gridCtx.fillStyle = '#10b981';
            this.gridCtx.fillRect(goal[0] * cellSize, goal[1] * cellSize, cellSize, cellSize);
            
            // Contour des goals
            this.gridCtx.strokeStyle = '#059669';
            this.gridCtx.lineWidth = 2;
            this.gridCtx.strokeRect(goal[0] * cellSize, goal[1] * cellSize, cellSize, cellSize);
        });

        // Draw agent path
        if (positions && positions.length > 0) {
            // Draw trajectory
            this.gridCtx.strokeStyle = '#3b82f6';
            this.gridCtx.lineWidth = 3;
            this.gridCtx.beginPath();
            
            positions.forEach((pos, index) => {
                const x = pos[0] * cellSize + cellSize / 2;
                const y = pos[1] * cellSize + cellSize / 2;
                
                if (index === 0) {
                    this.gridCtx.moveTo(x, y);
                } else {
                    this.gridCtx.lineTo(x, y);
                }
            });
            this.gridCtx.stroke();

            // Draw position dots
            positions.forEach((pos, index) => {
                const alpha = 0.3 + (index / positions.length) * 0.7;
                this.gridCtx.fillStyle = `rgba(59, 130, 246, ${alpha})`;
                this.gridCtx.beginPath();
                this.gridCtx.arc(
                    pos[0] * cellSize + cellSize / 2,
                    pos[1] * cellSize + cellSize / 2,
                    cellSize / 8, 0, 2 * Math.PI
                );
                this.gridCtx.fill();
            });

            // Draw current position
            const currentPos = positions[positions.length - 1];
            this.gridCtx.fillStyle = '#ef4444';
            this.gridCtx.beginPath();
            this.gridCtx.arc(
                currentPos[0] * cellSize + cellSize / 2,
                currentPos[1] * cellSize + cellSize / 2,
                cellSize / 3, 0, 2 * Math.PI
            );
            this.gridCtx.fill();
            
            // Contour position actuelle
            this.gridCtx.strokeStyle = '#dc2626';
            this.gridCtx.lineWidth = 2;
            this.gridCtx.stroke();
        }
    }

    drawError() {
        const { errors } = this.currentData;
        
        // Clear avec couleur de fond sombre
        this.errorCtx.fillStyle = '#0f172a';
        this.errorCtx.fillRect(0, 0, this.errorCanvas.width, this.errorCanvas.height);

        if (!errors || errors.length === 0) {
            // Message quand pas d'erreurs
            this.errorCtx.fillStyle = '#64748b';
            this.errorCtx.font = '16px Arial';
            this.errorCtx.textAlign = 'center';
            this.errorCtx.fillText('Training data will appear here', this.errorCanvas.width / 2, this.errorCanvas.height / 2);
            return;
        }

        // Draw grid
        this.errorCtx.strokeStyle = '#334155';
        this.errorCtx.lineWidth = 1;
        
        // Grid vertical
        for (let i = 0; i <= 10; i++) {
            const x = (i / 10) * this.errorCanvas.width;
            this.errorCtx.beginPath();
            this.errorCtx.moveTo(x, 0);
            this.errorCtx.lineTo(x, this.errorCanvas.height);
            this.errorCtx.stroke();
        }
        
        // Grid horizontal
        for (let i = 0; i <= 10; i++) {
            const y = (i / 10) * this.errorCanvas.height;
            this.errorCtx.beginPath();
            this.errorCtx.moveTo(0, y);
            this.errorCtx.lineTo(this.errorCanvas.width, y);
            this.errorCtx.stroke();
        }

        // Normalize errors
        const maxError = Math.max(...errors.filter(e => !isNaN(e) && isFinite(e)));
        const normalizedErrors = errors.map(e => e / (maxError || 1));

        // Draw curve with gradient
        const gradient = this.errorCtx.createLinearGradient(0, 0, this.errorCanvas.width, 0);
        gradient.addColorStop(0, '#3b82f6');
        gradient.addColorStop(1, '#8b5cf6');
        
        this.errorCtx.strokeStyle = gradient;
        this.errorCtx.lineWidth = 3;
        this.errorCtx.beginPath();

        normalizedErrors.forEach((error, index) => {
            const x = (index / normalizedErrors.length) * this.errorCanvas.width;
            const y = this.errorCanvas.height - (error * this.errorCanvas.height * 0.8) - 20;
            
            if (index === 0) {
                this.errorCtx.moveTo(x, y);
            } else {
                this.errorCtx.lineTo(x, y);
            }
        });
        this.errorCtx.stroke();

        // Draw points on significant episodes
        this.errorCtx.fillStyle = '#f59e0b';
        const step = Math.max(1, Math.floor(normalizedErrors.length / 10));
        for (let i = 0; i < normalizedErrors.length; i += step) {
            const x = (i / normalizedErrors.length) * this.errorCanvas.width;
            const y = this.errorCanvas.height - (normalizedErrors[i] * this.errorCanvas.height * 0.8) - 20;
            
            this.errorCtx.beginPath();
            this.errorCtx.arc(x, y, 4, 0, 2 * Math.PI);
            this.errorCtx.fill();
        }

        // Labels
        this.errorCtx.fillStyle = '#e2e8f0';
        this.errorCtx.font = 'bold 14px Arial';
        this.errorCtx.textAlign = 'center';
        this.errorCtx.fillText('Training Convergence', this.errorCanvas.width / 2, 20);
        
        this.errorCtx.font = '12px Arial';
        this.errorCtx.fillText('Episodes', this.errorCanvas.width / 2, this.errorCanvas.height - 5);
        
        this.errorCtx.save();
        this.errorCtx.translate(15, this.errorCanvas.height / 2);
        this.errorCtx.rotate(-Math.PI / 2);
        this.errorCtx.fillText('Error Value', 0, 0);
        this.errorCtx.restore();
    }

    startSimulation() {
        this.simulationRunning = true;
        this.updateTrainingStatus('Simulation running', 'training');
    }

    pauseSimulation() {
        this.simulationRunning = false;
        this.updateTrainingStatus('Simulation paused', 'ready');
    }

    async resetSimulation() {
        try {
            await fetch('/reset', { method: 'POST' });
            this.updateTrainingStatus('Ready to train', 'ready');
        } catch (error) {
            console.error('Error resetting simulation:', error);
            this.updateTrainingStatus('Reset error', 'error');
        }
    }
}

// Global functions for button clicks
function startSimulation() {
    if (window.rlApp) window.rlApp.startSimulation();
}

function pauseSimulation() {
    if (window.rlApp) window.rlApp.pauseSimulation();
}

function resetSimulation() {
    if (window.rlApp) window.rlApp.resetSimulation();
}

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    window.rlApp = new RLGridWorld();
});