document.addEventListener('DOMContentLoaded', () => {
    // --- 要素の取得 ---
    const mainView = document.getElementById('annotation-view');
    const progressIndicator = document.getElementById('progress-indicator');
    const btnUsed = document.getElementById('btn-used');
    const btnNotUsed = document.getElementById('btn-not-used');
    const btnNext = document.getElementById('btn-next');
    const btnCopyPrompt = document.getElementById('btn-copy-prompt');
    const btnSkipDataPaper = document.getElementById('btn-skip-datapaper');

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
            
            const pdfLinkHTML = paperData.pdf_url 
                ? `<a href="${paperData.pdf_url}" target="_blank">Open PDF in New Tab</a>`
                : '<span>No PDF link available</span>';

            mainView.innerHTML = `
                <div class="context-paper">
                    <h3>Data Paper (D): ${paperData.cited_datapaper_title}</h3>
                    <p><b>(Total 'Used' Candidates for this Data Paper: ${paperData.data_paper_total_candidates})</b></p>
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
        progressIndicator.textContent = `[${progressData.mode}] Progress: ${progressData.annotated} / ${progressData.total} (${percentage}%)`;
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
    
    async function handleSkipDataPaper() {
        if (!currentPaper) return;
        
        if (!confirm(`本当にこのデータ論文「${currentPaper.cited_datapaper_title}」の未確認候補をすべてスキップしますか？`)) {
            return;
        }

        btnSkipDataPaper.disabled = true;
        await fetch('/skip_datapaper', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                cited_datapaper_doi: currentPaper.cited_datapaper_doi
            })
        });
        
        await getNextTask();
        btnSkipDataPaper.disabled = false;
    }

    async function copyPrompt() {
        if (!currentPaper) return;

        try {
            const response = await fetch('/get_llm_prompt', {
                method: 'POST',
                // ▼▼▼ 修正点: 'ContentType' を 'Content-Type' に修正 ▼▼▼
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    cited_title: currentPaper.cited_datapaper_title,
                    citing_title: currentPaper.citing_paper_title,
                    citing_text: currentPaper.citing_paper_text
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to get prompt from server.');
            }

            const data = await response.json();
            
            navigator.clipboard.writeText(data.prompt).then(() => {
                console.log('Prompt copied to clipboard!');
            })
            .catch(err => {
                console.error('Failed to copy text: ', err);
                alert('コピーに失敗しました。\n(Error: ' + err.message + ')');
            });

        } catch (error) {
            console.error(error);
            alert('プロンプトの生成に失敗しました。');
        }
    }

    // --- イベントリスナーの設定 ---
    btnUsed.addEventListener('click', () => handleAnnotation('used'));
    btnNotUsed.addEventListener('click', () => handleAnnotation('not_used'));
    btnNext.addEventListener('click', getNextTask);
    btnCopyPrompt.addEventListener('click', copyPrompt);
    btnSkipDataPaper.addEventListener('click', handleSkipDataPaper);

    // --- 初期化 ---
    getNextTask();
});