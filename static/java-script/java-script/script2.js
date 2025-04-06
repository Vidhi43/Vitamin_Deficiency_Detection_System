// profile.js - JavaScript for profile page with delete functionality

document.addEventListener("DOMContentLoaded", function() {
    // DOM Elements
    const profileImage = document.getElementById('profileImage');
    const profileImageUpload = document.getElementById('profileImageUpload');
    const changePhotoBtn = document.getElementById('changePhotoBtn');
    const reportsList = document.getElementById('reportsList');
    const reportModal = document.getElementById('reportModal');
    const modalReportContent = document.getElementById('modalReportContent');
    const closeModal = document.querySelector('.close-modal');

    // Load saved reports from localStorage
    let reportsHistory = [];
    if (localStorage.getItem('reportsHistory')) {
        reportsHistory = JSON.parse(localStorage.getItem('reportsHistory'));
    }

    // Profile photo change functionality
    if (changePhotoBtn && profileImageUpload && profileImage) {
        changePhotoBtn.addEventListener('click', () => {
            profileImageUpload.click();
        });

        profileImageUpload.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    profileImage.src = e.target.result;
                    localStorage.setItem('profileImage', e.target.result);
                };
                reader.onerror = function() {
                    alert("Error reading the image file. Please try another image.");
                };
                reader.readAsDataURL(file);
            } else {
                alert("Please select a valid image file (JPEG, PNG, GIF).");
            }
        });

        // Load saved profile image if exists
        const savedImage = localStorage.getItem('profileImage');
        if (savedImage) {
            profileImage.src = savedImage;
        }
    }

    // Update reports list on page load
    updateReportsList();

    // Function to update reports list
    function updateReportsList() {
        if (!reportsList) return;
        
        if (reportsHistory.length === 0) {
            reportsList.innerHTML = '<div class="no-reports">No reports generated yet</div>';
            return;
        }

        reportsList.innerHTML = reportsHistory.map(report => `
            <div class="report-card" data-report-id="${report.id}">
                <div class="report-header">
                    <span class="report-title">${report.deficiency.name} Deficiency</span>
                    <span class="report-date">${report.date}</span>
                </div>
                <div class="report-details">
                    <span class="report-deficiency ${report.deficiency.severity.toLowerCase()}">
                        ${report.deficiency.severity} Severity
                    </span>
                    <div class="report-actions">
                        <button class="view-report-btn" data-report-id="${report.id}">
                            <i class="fas fa-eye"></i> View
                        </button>
                        <button class="delete-report-btn" data-report-id="${report.id}">
                            <i class="fas fa-trash"></i> Delete
                        </button>
                    </div>
                </div>
            </div>
        `).join('');

        // Add event listeners to view buttons
        document.querySelectorAll('.view-report-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const reportId = e.target.closest('button').getAttribute('data-report-id');
                const report = reportsHistory.find(r => r.id === reportId);
                if (report) {
                    showReportModal(report);
                }
            });
        });

        // Add event listeners to delete buttons
        document.querySelectorAll('.delete-report-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const reportId = e.target.closest('button').getAttribute('data-report-id');
                deleteReport(reportId);
            });
        });
    }

    // Function to delete a report
    function deleteReport(reportId) {
        if (confirm('Are you sure you want to delete this report? This action cannot be undone.')) {
            // Filter out the report to be deleted
            reportsHistory = reportsHistory.filter(report => report.id !== reportId);
            
            // Update localStorage
            localStorage.setItem('reportsHistory', JSON.stringify(reportsHistory));
            
            // Update the UI
            updateReportsList();
            
            // Show feedback
            alert('Report deleted successfully');
        }
    }

    // Function to show report modal
    // Function to show report modal
function showReportModal(report) {
    modalReportContent.innerHTML = `
        <div id="reportToPrint" class="doctor-report" style="width: 100%;">
            <div style="margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <div>
                        <h1 style="color: #4c97a3; margin: 0 0 5px 0;">Vitamin Deficiency Detection</h1>
                        <h2 style="color: #333; margin: 0 0 15px 0;">Medical Report</h2>
                    </div>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div class="patient-info" style="flex: 1;">
                        <p style="margin: 5px 0;"><strong>Patient Name:</strong> ${document.querySelector('.profile p') ? document.querySelector('.profile p').textContent.split(':')[1].trim() : 'User'}</p>
                        <p style="margin: 5px 0;"><strong>Date:</strong> ${report.date}</p>
                        <p style="margin: 5px 0;"><strong>Time:</strong> ${report.time}</p>
                        <p style="margin: 5px 0;"><strong>Report ID:</strong> ${report.id}</p>
                    </div>
                    
                    <div class="image-container" style="flex: 0 0 300px; margin-left: 30px;">
                        <img src="${report.image}" alt="Analyzed image" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px;">
                    </div>
                </div>
            </div>

            <div class="results-container" style="margin-top: 20px;">
                <h3 style="color: #4c97a3;">${report.deficiency.name} Deficiency</h3>
                <div class="severity ${report.deficiency.severity.toLowerCase()}" style="padding: 5px 10px; display: inline-block; border-radius: 3px; background-color: ${report.deficiency.severity === 'Severe' ? '#ffcccc' : report.deficiency.severity === 'Moderate' ? '#ffe6cc' : '#e6ffe6'};">
                    ${report.deficiency.severity} severity
                    <span class="confidence">(${report.deficiency.confidence}% confidence)</span>
                </div>

                <div class="symptoms" style="margin-top: 15px;">
                    <h4 style="color: #4c97a3;">Symptoms:</h4>
                    <ul style="margin-left: 20px;">${report.deficiency.symptoms.map(s => `<li style="margin-bottom: 5px;">${s}</li>`).join('')}</ul>
                </div>

                <div class="causes" style="margin-top: 15px;">
                    <h4 style="color: #4c97a3;">Possible Causes:</h4>
                    <ul style="margin-left: 20px;">${report.deficiency.causes.map(c => `<li style="margin-bottom: 5px;">${c}</li>`).join('')}</ul>
                </div>

                <div class="recommendations" style="margin-top: 15px;">
                    <h4 style="color: #4c97a3;">Recommendations:</h4>
                    <ul style="margin-left: 20px;">${report.deficiency.recommendations.map(r => `<li style="margin-bottom: 5px;">${r}</li>`).join('')}</ul>
                </div>
            </div>

            <div class="report-footer" style="margin-top: 30px; font-size: 12px; color: #666; border-top: 1px solid #eee; padding-top: 10px;">
                <p><strong>Disclaimer:</strong> This report is generated by AI analysis and should be reviewed by a medical professional.</p>
            </div>
        </div>

        <div class="modal-actions" style="margin-top: 20px;">
            <button class="download-btn" onclick="downloadReport(${JSON.stringify(report).replace(/"/g, '&quot;')})">
                <i class="fas fa-file-pdf"></i> Download PDF
            </button>
            <button class="delete-btn" onclick="deleteReportFromModal('${report.id}')">
                <i class="fas fa-trash"></i> Delete Report
            </button>
        </div>
    `;
    
    reportModal.style.display = 'flex';
}
    

    // Global function for deleting report from modal
    window.deleteReportFromModal = function(reportId) {
        reportModal.style.display = 'none';
        deleteReport(reportId);
    };

    // Close modal when clicking X
    if (closeModal) {
        closeModal.addEventListener('click', () => {
            reportModal.style.display = 'none';
        });
    }

    // Close modal when clicking outside
    reportModal.addEventListener('click', (e) => {
        if (e.target === reportModal) {
            reportModal.style.display = 'none';
        }
    });

    // Global function for downloading report (used in modal)
    window.downloadReport = async function(report) {
        try {
            // Show loading state
            const downloadBtn = document.querySelector('.download-btn');
            if (downloadBtn) {
                downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating PDF...';
                downloadBtn.disabled = true;
            }
    
            // Create a clone of the element to modify for PDF
            const element = document.getElementById('reportToPrint');
            if (!element) return;
            
            const elementClone = element.cloneNode(true);
            document.body.appendChild(elementClone);
            
            // Apply PDF-specific styles
            elementClone.style.width = '794px'; // A4 width in pixels (210mm)
            elementClone.style.padding = '20px';
            elementClone.style.margin = '0';
            elementClone.style.boxSizing = 'border-box';
            elementClone.style.fontSize = '12px';
            elementClone.style.position = 'fixed';
            elementClone.style.left = '0';
            elementClone.style.top = '0';
            elementClone.style.visibility = 'visible';
            elementClone.style.zIndex = '9999';
            elementClone.style.backgroundColor = '#FFFFFF';
    
            // Modify content to be more compact
            const header = elementClone.querySelector('h1');
            if (header) header.style.fontSize = '20px';
            
            const h2 = elementClone.querySelector('h2');
            if (h2) h2.style.fontSize = '16px';
            
            const h3 = elementClone.querySelector('h3');
            if (h3) h3.style.fontSize = '14px';
            
            const h4s = elementClone.querySelectorAll('h4');
            h4s.forEach(h4 => h4.style.fontSize = '13px');
            
            const lists = elementClone.querySelectorAll('ul');
            lists.forEach(ul => {
                ul.style.marginLeft = '15px';
                ul.style.paddingLeft = '5px';
            });
            
            const listItems = elementClone.querySelectorAll('li');
            listItems.forEach(li => {
                li.style.marginBottom = '3px';
                li.style.lineHeight = '1.3';
            });
    
            // Reduce image size if needed
            const img = elementClone.querySelector('img');
            if (img) {
                img.style.maxWidth = '250px';
                img.style.height = 'auto';
            }
    
            // PDF settings
            const { jsPDF } = window.jspdf;
            const pdf = new jsPDF('p', 'mm', 'a4');
            const pageWidth = pdf.internal.pageSize.getWidth();
            const pageHeight = pdf.internal.pageSize.getHeight();
    
            // Capture with html2canvas
            const canvas = await html2canvas(elementClone, {
                scale: 2,
                logging: false,
                useCORS: true,
                backgroundColor: '#FFFFFF',
                scrollX: 0,
                scrollY: 0,
                windowWidth: 794, // Fixed width for A4
                windowHeight: elementClone.scrollHeight,
                ignoreElements: (element) => {
                    // Ignore elements that might cause overflow
                    return element.classList && element.classList.contains('modal-actions');
                }
            });
    
            // Calculate image dimensions to fit page
            const imgWidth = pageWidth - 20; // 10mm margins
            const imgHeight = (canvas.height * imgWidth) / canvas.width;
    
            // Scale down if content is too tall
            let finalHeight = imgHeight;
            if (imgHeight > pageHeight - 20) {
                const scaleFactor = (pageHeight - 20) / imgHeight;
                finalHeight = (pageHeight - 20);
                pdf.addImage(
                    canvas.toDataURL('image/png'),
                    'PNG',
                    10,
                    10,
                    imgWidth * scaleFactor,
                    finalHeight
                );
            } else {
                pdf.addImage(
                    canvas.toDataURL('image/png'),
                    'PNG',
                    10,
                    10,
                    imgWidth,
                    finalHeight
                );
            }
    
            // Add footer
            pdf.setFontSize(10);
            pdf.setTextColor(100);
            pdf.text('Generated by Vitamin Deficiency Detector', pageWidth/2, pageHeight - 10, {align: 'center'});
    
            // Save and clean up
            pdf.save(`Vitamin_Report_${report.id}.pdf`);
            document.body.removeChild(elementClone);
    
        } catch (error) {
            console.error("PDF Error:", error);
            alert("Failed to generate PDF. See console for details.");
        } finally {
            const downloadBtn = document.querySelector('.download-btn');
            if (downloadBtn) {
                downloadBtn.innerHTML = '<i class="fas fa-file-pdf"></i> Download PDF';
                downloadBtn.disabled = false;
            }
        }
    };
});