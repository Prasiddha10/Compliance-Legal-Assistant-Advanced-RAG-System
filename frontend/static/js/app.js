// Global variables
let currentTab = 'chat';
let isLoading = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

async function initializeApp() {
    try {
        // Check system health
        const healthCheck = await fetch('http://localhost:8004/health');
        const healthData = await healthCheck.json();
        
        if (!healthData.system_initialized) {
            console.warn('System not fully initialized');
        }
        
        // Initialize event listeners
        setupEventListeners();
        
        // Load initial stats
        await refreshStats();
        
        // Hide loading screen
        setTimeout(() => {
            document.getElementById('loading-screen').classList.add('hidden');
        }, 1000);
        
    } catch (error) {
        console.error('Error initializing app:', error);
        showError('Failed to initialize application. Please refresh the page.');
    }
}

function setupEventListeners() {
    // Chat input enter key
    document.getElementById('query-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });
    
    // File upload
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('upload-area');
    
    fileInput.addEventListener('change', handleFileUpload);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        handleFileUpload({ target: { files } });
    });
    
    // Auto-refresh stats every 30 seconds
    setInterval(refreshStats, 30000);
}

// Tab Management
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Add active class to selected nav button
    event.target.classList.add('active');
    
    currentTab = tabName;
    
    // Load tab-specific data
    if (tabName === 'stats') {
        refreshStats();
    }
}

// Chat Functions
async function sendQuery() {
    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');
    const languageSelect = document.getElementById('language-select');
    const modelSelect = document.getElementById('model-select');
    
    const query = queryInput.value.trim();
    if (!query || isLoading) return;
    
    // Update UI
    isLoading = true;
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    
    // Add user message
    addMessage(query, 'user');
    queryInput.value = '';
    
    try {
        const response = await fetch('http://localhost:8004/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                language: languageSelect.value,
                model: modelSelect.value
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Add bot response
        addMessage(data.answer, 'bot', data.sources, data);
        
    } catch (error) {
        console.error('Error sending query:', error);
        addMessage('Sorry, I encountered an error while processing your query. Please try again.', 'bot');
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
    }
}

function addMessage(text, sender, sources = null, metadata = null) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message fade-in`;
    
    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="message-sources">
                <h4><i class="fas fa-book"></i> Sources:</h4>
                ${sources.map(source => `
                    <div class="source-item">
                        <div class="source-title">${source.metadata.source || 'Unknown Source'}</div>
                        <div class="source-content">${source.content.substring(0, 200)}...</div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    let evaluationHtml = '';
    if (metadata && metadata.evaluation_scores && sender === 'bot') {
        const scores = metadata.evaluation_scores;
        evaluationHtml = `
            <div class="message-evaluations">
                <h4><i class="fas fa-chart-line"></i> Evaluation Scores:</h4>
                <div class="evaluation-scores">
                    ${scores.overall_score !== undefined ? `
                        <div class="score-item">
                            <span class="score-label">Overall</span>
                            <span class="score-value ${getScoreClass(scores.overall_score)}">${scores.overall_score.toFixed(2)}</span>
                        </div>
                    ` : ''}
                    ${scores.generation_score !== undefined ? `
                        <div class="score-item">
                            <span class="score-label">Generation</span>
                            <span class="score-value ${getScoreClass(scores.generation_score)}">${scores.generation_score.toFixed(2)}</span>
                        </div>
                    ` : ''}
                    ${scores.judge_score !== undefined ? `
                        <div class="score-item">
                            <span class="score-label">LLM Judge</span>
                            <span class="score-value ${getScoreClass(scores.judge_score)}">${scores.judge_score.toFixed(2)}</span>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }
    
    let metadataHtml = '';
    if (metadata && sender === 'bot') {
        let languageInfo = '';
        if (metadata.language_used && metadata.detected_language) {
            const languageDisplay = metadata.language_used === 'auto' ? 
                `Auto-detected: ${metadata.detected_language.toUpperCase()}` : 
                `Language: ${metadata.language_used.toUpperCase()}`;
            
            languageInfo = ` • ${languageDisplay}`;
        }
        
        metadataHtml = `
            <div class="message-metadata">
                <small style="color: #666; font-size: 0.8rem;">
                    ${metadata.model_used} • ${metadata.database_used} • ${metadata.processing_time.toFixed(2)}s${languageInfo}
                </small>
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="message-text">${formatMessage(text)}</div>
            ${sourcesHtml}
            ${evaluationHtml}
            ${metadataHtml}
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function getScoreClass(score) {
    if (score >= 0.8) return 'excellent';
    if (score >= 0.6) return 'good';
    if (score >= 0.4) return 'fair';
    return 'poor';
}

function formatMessage(text) {
    // Convert markdown-style formatting to HTML
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    text = text.replace(/\n/g, '<br>');
    
    // Convert article references to highlighted text
    text = text.replace(/Article\s+(\d+)/gi, '<span style="background: #667eea; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold;">Article $1</span>');
    
    // Convert section references to highlighted text
    text = text.replace(/Section\s+(\d+)/gi, '<span style="background: #2ed573; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold;">Section $1</span>');
    
    return text;
}

// Upload Functions
async function handleFileUpload(event) {
    const files = event.target.files;
    if (!files.length) return;
    
    const uploadProgress = document.getElementById('upload-progress');
    const uploadStatus = document.getElementById('upload-status');
    const progressFill = document.getElementById('progress-fill');
    
    uploadProgress.style.display = 'block';
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        // Check file type
        if (!file.type.includes('pdf')) {
            showError(`File ${file.name} is not a PDF. Only PDF files are supported.`);
            continue;
        }
        
        try {
            uploadStatus.textContent = `Uploading ${file.name}...`;
            progressFill.style.width = '0%';
            
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('http://localhost:8004/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            progressFill.style.width = '100%';
            uploadStatus.textContent = `${file.name} uploaded successfully!`;
            
            showSuccess(`File ${file.name} uploaded and is being processed.`);
            
        } catch (error) {
            console.error('Error uploading file:', error);
            showError(`Failed to upload ${file.name}. Please try again.`);
        }
    }
    
    // Hide progress after 3 seconds
    setTimeout(() => {
        uploadProgress.style.display = 'none';
        document.getElementById('file-input').value = '';
    }, 3000);
}

// Evaluation Functions
function addQuery() {
    const queryInputs = document.querySelector('.query-inputs');
    const newQueryDiv = document.createElement('div');
    newQueryDiv.className = 'query-input-group';
    newQueryDiv.innerHTML = `
        <input type="text" class="eval-query" placeholder="Enter test query...">
        <button onclick="removeQuery(this)"><i class="fas fa-trash"></i></button>
    `;
    queryInputs.appendChild(newQueryDiv);
}

function removeQuery(button) {
    button.parentElement.remove();
}

async function runEvaluation() {
    const queryInputs = document.querySelectorAll('.eval-query');
    const detailedMetrics = document.getElementById('detailed-metrics').checked;
    
    const queries = Array.from(queryInputs)
        .map(input => input.value.trim())
        .filter(query => query.length > 0);
    
    if (queries.length === 0) {
        showError('Please enter at least one query for evaluation.');
        return;
    }
    
    const evaluateBtn = document.querySelector('.evaluate-btn');
    evaluateBtn.disabled = true;
    evaluateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running Evaluation...';
    
    try {
        const response = await fetch('http://localhost:8004/api/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                queries: queries,
                include_detailed_metrics: detailedMetrics
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        displayEvaluationResults(data);
        
    } catch (error) {
        console.error('Error running evaluation:', error);
        showError('Failed to run evaluation. Please try again.');
    } finally {
        evaluateBtn.disabled = false;
        evaluateBtn.innerHTML = '<i class="fas fa-play"></i> Run Evaluation';
    }
}

function displayEvaluationResults(data) {
    const resultsContainer = document.getElementById('evaluation-results');
    const summaryContainer = document.getElementById('results-summary');
    const detailsContainer = document.getElementById('results-details');
    
    // Display summary
    summaryContainer.innerHTML = `
        <div class="result-summary">
            <h4>Evaluation Summary</h4>
            <p><strong>Total Queries:</strong> ${data.summary.total_queries}</p>
            <p><strong>Average Score:</strong> ${data.summary.average_score.toFixed(2)}</p>
            <p><strong>Completed:</strong> ${new Date(data.summary.timestamp * 1000).toLocaleString()}</p>
        </div>
    `;
    
    // Display detailed results
    detailsContainer.innerHTML = data.evaluation_results.map(result => `
        <div class="result-item">
            <div class="result-query">${result.query}</div>
            <div class="result-score">Overall Score: ${result.overall_score?.toFixed(2) || 'N/A'}</div>
            ${result.detailed_metrics ? `
                <div class="result-details">
                    <p><strong>Retrieval Score:</strong> ${result.detailed_metrics.retrieval_score?.toFixed(2) || 'N/A'}</p>
                    <p><strong>Generation Score:</strong> ${result.detailed_metrics.generation_score?.toFixed(2) || 'N/A'}</p>
                    <p><strong>Processing Time:</strong> ${result.processing_time?.toFixed(2) || 'N/A'}s</p>
                </div>
            ` : ''}
        </div>
    `).join('');
    
    resultsContainer.style.display = 'block';
}

// Stats Functions
async function refreshStats() {
    try {
        const response = await fetch('http://localhost:8004/api/stats');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update system status
        document.getElementById('system-status').textContent = 
            data.system_status === 'operational' ? '✅ Operational' : '❌ Offline';
        
        // Update ChromaDB status
        const chromaStatus = data.databases.chroma;
        document.getElementById('chroma-status').innerHTML = chromaStatus.status === 'connected' 
            ? `✅ Connected<br><small>${chromaStatus.document_count} documents</small>`
            : `❌ ${chromaStatus.status}<br><small>${chromaStatus.error || 'Unknown error'}</small>`;
        
        // Update Pinecone status
        const pineconeStatus = data.databases.pinecone;
        document.getElementById('pinecone-status').innerHTML = pineconeStatus.status === 'connected'
            ? `✅ Connected<br><small>${pineconeStatus.document_count} documents</small>`
            : `❌ ${pineconeStatus.status}<br><small>${pineconeStatus.error || 'Unknown error'}</small>`;
        
    } catch (error) {
        console.error('Error refreshing stats:', error);
        document.getElementById('system-status').textContent = '❌ Error';
        document.getElementById('chroma-status').textContent = '❌ Error';
        document.getElementById('pinecone-status').textContent = '❌ Error';
    }
}

// Utility Functions
function showError(message) {
    showNotification(message, 'error');
}

function showSuccess(message) {
    showNotification(message, 'success');
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        z-index: 10000;
        animation: slideInRight 0.3s ease;
        max-width: 400px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    `;
    
    // Set background color based on type
    switch (type) {
        case 'error':
            notification.style.background = 'linear-gradient(135deg, #ff4757, #ff3838)';
            break;
        case 'success':
            notification.style.background = 'linear-gradient(135deg, #2ed573, #26d063)';
            break;
        default:
            notification.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
    }
    
    notification.textContent = message;
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 5000);
}

// Add CSS animations for notifications
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(notificationStyles);

// Initialize tab navigation
document.addEventListener('DOMContentLoaded', function() {
    // Set initial active tab
    document.querySelector('.nav-btn').classList.add('active');
    
    // Add click handlers to navigation buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            showTab(this.textContent.trim().toLowerCase());
        });
    });
});
