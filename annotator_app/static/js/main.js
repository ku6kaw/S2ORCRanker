document.addEventListener('DOMContentLoaded', () => {
    // --- 要素の取得 ---
    const mainView = document.getElementById('annotation-view');
    const progressIndicator = document.getElementById('progress-indicator');
    const btnUsed = document.getElementById('btn-used');
    const btnNotUsed = document.getElementById('btn-not-used');
    const btnNext = document.getElementById('btn-next');
    const btnCopyPrompt = document.getElementById('btn-copy-prompt');

    // --- 状態管理 ---
    let currentPaper = null;

    // --- 関数定義 ---
    function updateView(paperData, progressData) {
        currentPaper = paperData;
        updateProgress(progressData);

        if (paperData) {
            const llmStatus = paperData.llm_annotation_status;
            const llmSuggestionHTML = llmStatus === 1 ? '<span class="suggestion-used">Used</span>' :
                                      llmStatus === -1 ? '<span class="suggestion-not-used">Not Used</span>' :
                                      '<span>Unprocessed</span>';
            
            // ▼▼▼ 修正点: PDFリンクがあればリンクを、なければメッセージを作成 ▼▼▼
            const pdfLinkHTML = paperData.pdf_url 
                ? `<a href="${paperData.pdf_url}" target="_blank">Open PDF in New Tab</a>`
                : '<span>No PDF link available</span>';

            mainView.innerHTML = `
                <div class="context-paper">
                    <h3>Data Paper (D): ${paperData.cited_datapaper_title}</h3>
                </div>
                <hr>
                <div class="candidate-paper">
                    <h4>Candidate: ${paperData.citing_paper_title}</h4>
                    <p>
                        <b>AI's Suggestion:</b> ${llmSuggestionHTML} | 
                        <b>PDF Link:</b> ${pdfLinkHTML}
                    </p>
                    <textarea readonly>${paperData.citing_paper_text}</textarea>
                </div>
            `;
        } else {
            mainView.innerHTML = '<h2>🎉 全ての論文のアノテーションが完了しました！</h2>';
            document.querySelector('footer').style.display = 'none';
        }
    }

    function updateProgress(progressData) {
        const percentage = progressData.total > 0 ? (progressData.annotated / progressData.total * 100).toFixed(1) : 0;
        progressIndicator.textContent = `Progress: ${progressData.annotated} / ${progressData.total} (${percentage}%)`;
    }

    async function getNextTask() {
        try {
            const response = await fetch('/get_task');
            const data = await response.json();
            updateView(data.paper, data.progress);
        } catch (error) {
            mainView.innerHTML = '<p style="color: red;">Error: Could not connect to the server.</p>';
        }
    }

    async function handleAnnotation(decision) {
        if (!currentPaper) return;

        btnUsed.disabled = true;
        btnNotUsed.disabled = true;

        await fetch('/annotate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                decision: decision,
                citing_doi: currentPaper.citing_doi,
                cited_datapaper_doi: currentPaper.cited_datapaper_doi
            })
        });

        await getNextTask();
        
        btnUsed.disabled = false;
        btnNotUsed.disabled = false;
    }
    
    function copyPrompt() {
        if (!currentPaper) return;
        const promptText = `この論文の要点を3行でまとめて。\n\nTitle: ${currentPaper.citing_paper_title}\n\nText: ${currentPaper.citing_paper_text.substring(0, 2000)}`;
        navigator.clipboard.writeText(promptText).then(() => {
            alert('Prompt copied to clipboard!');
        });
    }

    // --- イベントリスナーの設定 ---
    btnUsed.addEventListener('click', () => handleAnnotation('used'));
    btnNotUsed.addEventListener('click', () => handleAnnotation('not_used'));
    btnNext.addEventListener('click', getNextTask);
    btnCopyPrompt.addEventListener('click', copyPrompt);

    // --- 初期化 ---
    getNextTask();
});