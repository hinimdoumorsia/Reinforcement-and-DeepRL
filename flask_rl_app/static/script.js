class RLGridWorld {
    constructor() {
        this.gridCanvas = document.getElementById("gridCanvas");
        this.gridCtx = this.gridCanvas.getContext("2d");
        this.errorCanvas = document.getElementById("errorCanvas");
        this.errorCtx = this.errorCanvas.getContext("2d");
        
        // NOUVEAUX CANVAS
        this.policyCanvas = document.getElementById("policyCanvas");
        this.policyCtx = this.policyCanvas?.getContext("2d");
        this.rewardCanvas = document.getElementById("rewardCanvas");
        this.rewardCtx = this.rewardCanvas?.getContext("2d");
        
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
        
        // NOUVEAUX STYLES
        if (this.policyCanvas) this.policyCanvas.style.background = '#0f172a';
        if (this.rewardCanvas) this.rewardCanvas.style.background = '#0f172a';
    }

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

    // MISE À JOUR DE updateDisplay() AVEC NOUVELLES VISUALISATIONS
    updateDisplay() {
        if (!this.currentData) return;

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

        // Draw ALL visualizations
        this.drawGrid();
        this.drawError();
        this.drawPolicy();
        this.drawRewardEvolution();
        this.updateStats();
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

    // NOUVELLE MÉTHODE POUR LA POLITIQUE
    drawPolicy() {
        if (!this.currentData || !this.policyCtx) return;
        
        const { size, agent_name, goals, obstacles } = this.currentData;
        const cellSize = this.policyCanvas.width / size;
        
        // Clear canvas
        this.policyCtx.fillStyle = '#0f172a';
        this.policyCtx.fillRect(0, 0, this.policyCanvas.width, this.policyCanvas.height);
        
        // Dessiner la grille
        this.policyCtx.strokeStyle = '#334155';
        this.policyCtx.lineWidth = 1;
        for (let i = 0; i <= size; i++) {
            this.policyCtx.beginPath();
            this.policyCtx.moveTo(i * cellSize, 0);
            this.policyCtx.lineTo(i * cellSize, this.policyCanvas.height);
            this.policyCtx.stroke();
            
            this.policyCtx.beginPath();
            this.policyCtx.moveTo(0, i * cellSize);
            this.policyCtx.lineTo(this.policyCanvas.width, i * cellSize);
            this.policyCtx.stroke();
        }
        
        // Dessiner obstacles
        obstacles.forEach(obs => {
            this.policyCtx.fillStyle = '#475569';
            this.policyCtx.fillRect(obs[0] * cellSize, obs[1] * cellSize, cellSize, cellSize);
        });

        // Dessiner goals
        goals.forEach(goal => {
            this.policyCtx.fillStyle = '#10b981';
            this.policyCtx.fillRect(goal[0] * cellSize, goal[1] * cellSize, cellSize, cellSize);
        });

        // Dessiner des flèches de politique basiques (exemple)
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // Éviter les obstacles et goals
                if (obstacles.some(obs => obs[0] === x && obs[1] === y) ||
                    goals.some(goal => goal[0] === x && goal[1] === y)) {
                    continue;
                }
                
                // Flèche simple pointant vers la droite (à améliorer)
                this.drawArrow(this.policyCtx, x, y, cellSize, 'right');
            }
        }
        
        // Labels
        this.policyCtx.fillStyle = '#e2e8f0';
        this.policyCtx.font = 'bold 14px Arial';
        this.policyCtx.textAlign = 'center';
        this.policyCtx.fillText(`Policy: ${agent_name}`, this.policyCanvas.width/2, 20);
    }

    // NOUVELLE MÉTHODE POUR DESSINER LES FLÈCHES
    drawArrow(ctx, x, y, cellSize, direction) {
        const centerX = x * cellSize + cellSize / 2;
        const centerY = y * cellSize + cellSize / 2;
        const arrowSize = cellSize / 4;
        
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 2;
        ctx.fillStyle = '#3b82f6';
        
        ctx.beginPath();
        
        switch(direction) {
            case 'up':
                ctx.moveTo(centerX, centerY - arrowSize);
                ctx.lineTo(centerX - arrowSize/2, centerY);
                ctx.lineTo(centerX + arrowSize/2, centerY);
                break;
            case 'right':
                ctx.moveTo(centerX + arrowSize, centerY);
                ctx.lineTo(centerX, centerY - arrowSize/2);
                ctx.lineTo(centerX, centerY + arrowSize/2);
                break;
            case 'down':
                ctx.moveTo(centerX, centerY + arrowSize);
                ctx.lineTo(centerX - arrowSize/2, centerY);
                ctx.lineTo(centerX + arrowSize/2, centerY);
                break;
            case 'left':
                ctx.moveTo(centerX - arrowSize, centerY);
                ctx.lineTo(centerX, centerY - arrowSize/2);
                ctx.lineTo(centerX, centerY + arrowSize/2);
                break;
        }
        
        ctx.closePath();
        ctx.fill();
    }

    // NOUVELLE MÉTHODE POUR L'ÉVOLUTION DES REWARDS
    drawRewardEvolution() {
        if (!this.currentData || !this.rewardCtx) return;
        
        const { errors } = this.currentData;
        
        // Clear canvas
        this.rewardCtx.fillStyle = '#0f172a';
        this.rewardCtx.fillRect(0, 0, this.rewardCanvas.width, this.rewardCanvas.height);
        
        if (!errors || errors.length === 0) {
            this.rewardCtx.fillStyle = '#64748b';
            this.rewardCtx.font = '14px Arial';
            this.rewardCtx.textAlign = 'center';
            this.rewardCtx.fillText('Reward data will appear here', this.rewardCanvas.width/2, this.rewardCanvas.height/2);
            return;
        }
        
        // Calculer les rewards (simplifié)
        const rewards = errors.map((error, index) => {
            // Reward augmente quand l'erreur diminue
            return Math.max(0, 1 - error) * 10;
        });

        // Normaliser les rewards
        const maxReward = Math.max(...rewards.filter(r => !isNaN(r) && isFinite(r)));
        const normalizedRewards = rewards.map(r => r / (maxReward || 1));

        // Dessiner la courbe des rewards
        const gradient = this.rewardCtx.createLinearGradient(0, 0, this.rewardCanvas.width, 0);
        gradient.addColorStop(0, '#10b981');
        gradient.addColorStop(1, '#f59e0b');
        
        this.rewardCtx.strokeStyle = gradient;
        this.rewardCtx.lineWidth = 3;
        this.rewardCtx.beginPath();

        normalizedRewards.forEach((reward, index) => {
            const x = (index / normalizedRewards.length) * this.rewardCanvas.width;
            const y = this.rewardCanvas.height - (reward * this.rewardCanvas.height * 0.8) - 20;
            
            if (index === 0) {
                this.rewardCtx.moveTo(x, y);
            } else {
                this.rewardCtx.lineTo(x, y);
            }
        });
        this.rewardCtx.stroke();

        // Labels
        this.rewardCtx.fillStyle = '#e2e8f0';
        this.rewardCtx.font = 'bold 14px Arial';
        this.rewardCtx.textAlign = 'center';
        this.rewardCtx.fillText('Reward Evolution', this.rewardCanvas.width / 2, 20);
        
        this.rewardCtx.font = '12px Arial';
        this.rewardCtx.fillText('Episodes', this.rewardCanvas.width / 2, this.rewardCanvas.height - 5);
    }

    // NOUVELLE MÉTHODE POUR LES STATISTIQUES
    updateStats() {
        if (!this.currentData) return;
        
        const { current_episode, total_episodes, training_complete, positions = [] } = this.currentData;
        
        // Mettre à jour les statistiques
        const currentEpisodeEl = document.getElementById('currentEpisode');
        const successRateEl = document.getElementById('successRate');
        
        if (currentEpisodeEl) {
            currentEpisodeEl.textContent = `${current_episode}/${total_episodes}`;
        }
        
        if (successRateEl) {
            let successRate = '0%';
            if (training_complete) {
                successRate = positions.length > 0 ? '100%' : '0%';
            } else if (current_episode > 0) {
                successRate = `${Math.min(100, (current_episode / total_episodes) * 100).toFixed(0)}%`;
            }
            successRateEl.textContent = successRate;
        }
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