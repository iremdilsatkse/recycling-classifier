document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const videoUpload = document.getElementById('video-upload');
    const submitBtn = document.querySelector('.submit-btn');
    const videoFeed = document.getElementById('video-feed');
    const statusDiv = document.getElementById('status');

    submitBtn.disabled = true;

    videoUpload.addEventListener('change', function() {
        submitBtn.disabled = !this.files.length;
    });

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        formData.append('file', videoUpload.files[0]);

        try {
            statusDiv.textContent = 'Uploading video...';
            statusDiv.className = 'status success';
            statusDiv.style.display = 'block';

            const response = await fetch('/video_upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Start video feed
            videoFeed.src = `/video_feed/${encodeURIComponent(data.video_path)}`;
            videoFeed.style.display = 'block';
            statusDiv.textContent = 'Processing video...';
            
            // Enable video feed error handling
            videoFeed.onerror = function() {
                statusDiv.textContent = 'Error processing video';
                statusDiv.className = 'status error';
            };

        } catch (error) {
            statusDiv.textContent = `Error: ${error.message}`;
            statusDiv.className = 'status error';
            statusDiv.style.display = 'block';
        }
    });
});