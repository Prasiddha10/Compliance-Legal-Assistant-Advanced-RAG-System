// Global variables
console.log('=== Compliance RAG System JS Loaded ===');
let currentTab = 'chat';
let isLoading = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

async function initializeApp() {
    try {
        // Check system health
        const healthCheck = await fetch('http://localhost:8005/health');
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
        const response = await fetch('http://localhost:8005/api/query', {
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
    
    let webSearchHtml = '';
    if (metadata && metadata.web_search_results && metadata.web_search_results.length > 0) {
        webSearchHtml = `
            <div class="message-web-search">
                <h4><i class="fas fa-globe"></i> Additional Web Resources:</h4>
                ${metadata.web_search_results.map(result => `
                    <div class="web-result-item">
                        <a href="${result.url}" target="_blank" class="web-result-link">
                            <strong>${result.title}</strong>
                        </a>
                        <p class="web-result-snippet">${result.snippet || 'No description available'}</p>
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
            ${webSearchHtml}
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

    // Format legal references with bold blue background style
    console.log('Original text before formatting:', text.substring(0, 200));
    
    // Create the bold blue styling like in official documents
    const articleStyle = 'background-color: #1976d2 !important; color: white !important; padding: 4px 8px !important; border-radius: 4px !important; border: 2px solid #0d47a1 !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; box-shadow: 0 2px 4px rgba(25, 118, 210, 0.3) !important; margin: 0 2px !important; display: inline-block !important;';
    
    // Format parenthetical article references like (Art.40), (Art.49(1)), (Art.55(2))
    text = text.replace(/\(Art\.(\d+(?:\(\d+\))?)\)/g, `<span class="article-reference" style="${articleStyle}">(ART.$1)</span>`);

    // Format standalone Art. references like Art.40, Art.49, Art.55
    text = text.replace(/\bArt\.(\d+(?:\(\d+\))?)\b/g, `<span class="article-reference" style="${articleStyle}">ART.$1</span>`);

    // Format Articles references like Articles 8-15, Article 55
    text = text.replace(/\bArticles?\s+(\d+(?:[-–]\d+)?)\b/g, `<span class="article-reference" style="${articleStyle}">ARTICLE $1</span>`);

    // Format section references
    text = text.replace(/Section\s+(\d+)/gi, `<span class="article-reference" style="${articleStyle}">SECTION $1</span>`);

    // Format document references like (Document 1)
    text = text.replace(/\((Document\s+\d+)\)/g, `<span class="article-reference" style="${articleStyle}">($1)</span>`);

    // Format recommendation references like (Rec.115)
    text = text.replace(/\((Rec\.\d+)\)/g, `<span class="article-reference" style="${articleStyle}">($1)</span>`);

    console.log('Formatted text with bold blue styling:', text.substring(0, 300));
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
        const allowedExtensions = ['.pdf', '.docx', '.txt'];
        const fileName = file.name.toLowerCase();
        const isValidFile = allowedExtensions.some(ext => fileName.endsWith(ext));
        
        if (!isValidFile) {
            showError(`File ${file.name} is not supported. Only PDF, DOCX, and TXT files are supported.`);
            continue;
        }
        
        try {
            // Clear any existing messages when starting new upload
            clearDuplicateMessage();
            clearSuccessMessage();

            uploadStatus.textContent = `Uploading ${file.name}...`;
            progressFill.style.width = '10%';

            // Start progress simulation for large files
            let progressInterval = setInterval(() => {
                let currentWidth = parseInt(progressFill.style.width) || 10;
                if (currentWidth < 90) {
                    progressFill.style.width = `${Math.min(currentWidth + 2, 90)}%`;
                    if (currentWidth < 30) {
                        uploadStatus.textContent = `Processing ${file.name}...`;
                    } else if (currentWidth < 60) {
                        uploadStatus.textContent = `Generating embeddings for ${file.name}...`;
                    } else {
                        uploadStatus.textContent = `Storing in databases for ${file.name}...`;
                    }
                }
            }, 500);
            
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('http://localhost:8005/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();

            // Clear progress interval and set to 100%
            clearInterval(progressInterval);
            progressFill.style.width = '100%';

            // Check for duplicate status
            if (data.is_duplicate) {
                // Extract database names from existing_in_databases
                const databaseNames = data.existing_in_databases ?
                    data.existing_in_databases.map(db => db.database || 'Unknown').join(', ') :
                    'ChromaDB, Pinecone';

                uploadStatus.textContent = `${file.name} already exists!`;
                showDuplicateMessage(`File "${file.name}" is already embedded in ${databaseNames}. No need to upload again.`);
            } else {
                uploadStatus.textContent = `${file.name} uploaded successfully!`;
                const successMessage = `File ${file.name} uploaded and is being processed. File hash: ${data.file_hash ? data.file_hash.substring(0, 8) + '...' : 'N/A'}`;
                showSuccess(successMessage);
                showSuccessMessage(successMessage);
                // Clear any existing duplicate message when a new file is uploaded
                clearDuplicateMessage();
            }
            
        } catch (error) {
            console.error('Error uploading file:', error);
            clearInterval(progressInterval);
            uploadStatus.textContent = `Failed to upload ${file.name}`;
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
    console.log('=== runEvaluation called ===');
    
    const queryInputs = document.querySelectorAll('.eval-query');
    console.log('Query inputs found:', queryInputs.length);
    
    const detailedMetrics = document.getElementById('detailed-metrics').checked;
    
    const queries = Array.from(queryInputs)
        .map(input => input.value.trim())
        .filter(query => query.length > 0);
    
    console.log('Valid queries:', queries);
    
    if (queries.length === 0) {
        showError('Please enter at least one query for evaluation.');
        return;
    }
    
    const evaluateBtn = document.querySelector('.evaluate-btn');
    if (!evaluateBtn) {
        console.error('Evaluate button not found!');
        showError('Evaluate button not found. Please refresh the page.');
        return;
    }
    
    evaluateBtn.disabled = true;
    evaluateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running Evaluation...';
    
    try {
        console.log('Starting evaluation with queries:', queries);
        console.log('Include detailed metrics:', detailedMetrics);
        
        const requestBody = {
            queries: queries,
            include_detailed_metrics: detailedMetrics
        };
        
        console.log('Request body:', JSON.stringify(requestBody, null, 2));
        
        const response = await fetch('http://localhost:8005/api/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        console.log('Response status:', response.status);
        console.log('Response ok:', response.ok);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Response error text:', errorText);
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }
        
        const data = await response.json();
        console.log('Evaluation response data:', data);
        
        displayEvaluationResults(data);
        
    } catch (error) {
        console.error('Error running evaluation:', error);
        
        let errorMessage = 'Failed to run evaluation';
        
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            errorMessage = 'Failed to connect to the backend server. Please ensure the backend is running on port 8005.';
        } else if (error.message.includes('HTTP error')) {
            errorMessage = `Server error: ${error.message}`;
        } else {
            errorMessage = `Failed to run evaluation: ${error.message}`;
        }
        
        showError(errorMessage);
    } finally {
        const evaluateBtn = document.querySelector('.evaluate-btn');
        if (evaluateBtn) {
            evaluateBtn.disabled = false;
            evaluateBtn.innerHTML = '<i class="fas fa-play"></i> Run Evaluation';
        }
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
                    <div class="metric-row">
                        <span class="metric-label">Retrieval Score:</span>
                        <span class="metric-value ${getScoreClass(result.detailed_metrics.retrieval_score || 0)}">${result.detailed_metrics.retrieval_score?.toFixed(2) || 'N/A'}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Generation Score:</span>
                        <span class="metric-value ${getScoreClass(result.detailed_metrics.generation_score || 0)}">${result.detailed_metrics.generation_score?.toFixed(2) || 'N/A'}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Judge Score:</span>
                        <span class="metric-value ${getScoreClass(result.detailed_metrics.judge_score || 0)}">${result.detailed_metrics.judge_score?.toFixed(2) || 'N/A'}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Processing Time:</span>
                        <span class="metric-value">${result.processing_time?.toFixed(2) || 'N/A'}s</span>
                    </div>
                    ${result.database_used ? `
                        <div class="metric-row">
                            <span class="metric-label">Database Used:</span>
                            <span class="metric-value">${result.database_used}</span>
                        </div>
                    ` : ''}
                </div>
            ` : ''}
        </div>
    `).join('');
    
    resultsContainer.style.display = 'block';
}

// Stats Functions
async function refreshStats() {
    try {
        const response = await fetch('http://localhost:8005/api/stats');
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

function showWarning(message) {
    showNotification(message, 'warning');
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
        case 'warning':
            notification.style.background = 'linear-gradient(135deg, #ffa502, #ff7675)';
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

    @keyframes slideInUp {
        from {
            transform: translateY(100%);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    @keyframes slideOutDown {
        from {
            transform: translateY(0);
            opacity: 1;
        }
        to {
            transform: translateY(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(notificationStyles);

// Function to show persistent duplicate message above footer
function showDuplicateMessage(message) {
    // Clear any existing messages
    clearDuplicateMessage();
    clearSuccessMessage();

    // Create duplicate message element
    const duplicateMessage = document.createElement('div');
    duplicateMessage.id = 'duplicate-message';
    duplicateMessage.style.cssText = `
        position: fixed;
        bottom: 60px;
        left: 20px;
        right: 20px;
        padding: 2rem 3rem;
        background: linear-gradient(135deg, #ffa502, #ff7675);
        color: white;
        font-weight: bold;
        text-align: center;
        z-index: 10000;
        animation: slideInUp 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        border: 3px solid #ff6b35;
        font-size: 1.0em;
        line-height: 1.4;
        max-width: 800px;
        margin: 0 auto;
        left: 50%;
        transform: translateX(-50%);
    `;

    duplicateMessage.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px; flex-wrap: wrap;">
            <span style="font-size: 0.95em; flex: 1; min-width: 300px;">${message}</span>
            <button onclick="clearDuplicateMessage()" style="
                background: rgba(255, 255, 255, 0.3);
                border: 2px solid rgba(255, 255, 255, 0.5);
                color: white;
                padding: 8px 12px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                font-size: 0.9em;
                transition: all 0.3s ease;
                min-width: 35px;
            " onmouseover="this.style.background='rgba(255, 255, 255, 0.4)'" onmouseout="this.style.background='rgba(255, 255, 255, 0.3)'">✕</button>
        </div>
    `;

    document.body.appendChild(duplicateMessage);
}

// Function to clear duplicate message
function clearDuplicateMessage() {
    const existingMessage = document.getElementById('duplicate-message');
    if (existingMessage) {
        existingMessage.style.animation = 'slideOutDown 0.3s ease';
        setTimeout(() => {
            if (existingMessage.parentNode) {
                existingMessage.parentNode.removeChild(existingMessage);
            }
        }, 300);
    }
}

// Function to show persistent success message above footer
function showSuccessMessage(message) {
    // Clear any existing messages
    clearSuccessMessage();
    clearDuplicateMessage();

    // Create success message element
    const successMessage = document.createElement('div');
    successMessage.id = 'success-message';
    successMessage.style.cssText = `
        position: fixed;
        bottom: 60px;
        left: 20px;
        right: 20px;
        padding: 2rem 3rem;
        background: linear-gradient(135deg, #2ed573, #26d063);
        color: white;
        font-weight: bold;
        text-align: center;
        z-index: 10000;
        animation: slideInUp 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        border: 3px solid #1dd1a1;
        font-size: 1.0em;
        line-height: 1.4;
        max-width: 800px;
        margin: 0 auto;
        left: 50%;
        transform: translateX(-50%);
    `;

    successMessage.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px; flex-wrap: wrap;">
            <span style="font-size: 0.95em; flex: 1; min-width: 300px;">${message}</span>
            <button onclick="clearSuccessMessage()" style="
                background: rgba(255, 255, 255, 0.3);
                border: 2px solid rgba(255, 255, 255, 0.5);
                color: white;
                padding: 8px 12px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                font-size: 0.9em;
                transition: all 0.3s ease;
                min-width: 35px;
            " onmouseover="this.style.background='rgba(255, 255, 255, 0.4)'" onmouseout="this.style.background='rgba(255, 255, 255, 0.3)'">✕</button>
        </div>
    `;

    document.body.appendChild(successMessage);
}

// Function to clear success message
function clearSuccessMessage() {
    const existingMessage = document.getElementById('success-message');
    if (existingMessage) {
        existingMessage.style.animation = 'slideOutDown 0.3s ease';
        setTimeout(() => {
            if (existingMessage.parentNode) {
                existingMessage.parentNode.removeChild(existingMessage);
            }
        }, 300);
    }
}

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
