document.addEventListener("DOMContentLoaded", function() {
    // DOM Elements
    const elements = {
        navbar: document.querySelector('.navbar'),
        hero: document.querySelector('.hero'),
        fadeElements: document.querySelectorAll('.fade-in'),
        uploadArea: document.getElementById('uploadArea'),
        imageUpload: document.getElementById('imageUpload'),
        imagePreview: document.getElementById('imagePreview'),
        uploadText: document.getElementById('uploadText'),
        uploadInstructions: document.getElementById('uploadInstructions'),
        detectBtn: document.getElementById('detectBtn'),
        resultsSection: document.getElementById('resultsSection'),
        resultsContent: document.getElementById('resultsContent')
    };

    // Current data
    let currentImageFile = null;
    let lastScroll = 0;

    // Initialize the application
    setupEventListeners();
    handleScroll();

    function handleScroll() {
        const currentScroll = window.pageYOffset;
        
        if (elements.navbar) {
            if (currentScroll > 50) {
                elements.navbar.classList.add('scrolled');
                if (elements.hero) elements.hero.classList.add('scrolled');
            } else {
                elements.navbar.classList.remove('scrolled');
                if (elements.hero) elements.hero.classList.remove('scrolled');
            }
        }
        lastScroll = currentScroll;
        animateOnScroll();
    }

    function animateOnScroll() {
        const windowHeight = window.innerHeight;
        const triggerPoint = 100;
        
        elements.fadeElements.forEach(el => {
            const elementPos = el.getBoundingClientRect().top;
            if (elementPos < windowHeight - triggerPoint) {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }
        });
    }

    function setupEventListeners() {
        // Fixed click handler for upload area
        elements.uploadArea.addEventListener('click', function(e) {
            if (e.target !== elements.imageUpload) {
                elements.imageUpload.click();
            }
        });

        // File selection handler
        elements.imageUpload.addEventListener('change', function(e) {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });

        // Drag and drop handlers
        elements.uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            elements.uploadArea.classList.add('dragover');
        });

        elements.uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            elements.uploadArea.classList.remove('dragover');
        });

        elements.uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            elements.uploadArea.classList.remove('dragover');
            
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileSelection(e.dataTransfer.files[0]);
            }
        });

        elements.detectBtn.addEventListener('click', analyzeImage);
        window.addEventListener('scroll', handleScroll);
    }

    function handleFileSelection(file) {
        if (!file.type.match('image.*')) {
            alert('Please upload an image file (JPEG, PNG, GIF)');
            return;
        }

        currentImageFile = file;
        
        elements.uploadText.textContent = file.name;
        elements.uploadInstructions.textContent = 'Click to change image';
        elements.detectBtn.disabled = false;

        const reader = new FileReader();
        reader.onload = function(e) {
            elements.imagePreview.src = e.target.result;
            elements.imagePreview.style.display = 'block';
        };
        reader.onerror = function() {
            alert('Error reading the image file. Please try another image.');
        };
        reader.readAsDataURL(file);
    }

    async function analyzeImage() {
        if (!currentImageFile) {
            alert('Please select an image first');
            return;
        }

        elements.detectBtn.disabled = true;
        elements.detectBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        
        try {
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            const result = {
                deficiencyType: ['A'],
                severity: 'Moderate',
                confidence: Math.floor(50 + Math.random() * 30)
            };
            
            const report = generateReport(result);
            displayResults(report);
            
        } catch (error) {
            console.error("Error analyzing image:", error);
            alert("Error analyzing image. Please try again.");
        } finally {
            elements.detectBtn.disabled = false;
            elements.detectBtn.textContent = 'Detect Deficiency';
        }
    }

    function generateReport(analysis) {
        const now = new Date();
        const deficiency = getDeficiencyInfo(analysis.deficiencyType[0]);
        
        return {
            id: 'RPT-' + Math.floor(10000 + Math.random() * 90000),
            date: now.toLocaleDateString(),
            time: now.toLocaleTimeString(),
            image: elements.imagePreview.src,
            deficiency: {
                name: deficiency.name,
                type: analysis.deficiencyType,
                severity: analysis.severity,
                confidence: analysis.confidence,
                symptoms: deficiency.symptoms,
                causes: deficiency.causes,
                recommendations: deficiency.recommendations
            },
            // Add compact HTML version for PDF
            compactHTML: `
           <div id="reportToPrint" class="doctor-report" style="width: 100%; font-family: Arial, sans-serif; padding: 15px; box-sizing: border-box;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 10px;">
                <div style="flex: 1;">
                    <h1 style="color: #4c97a3; margin: 0 0 5px 0; font-size: 18px;">Vitamin Deficiency Detection</h1>
                    <h2 style="color: #333; margin: 0 0 10px 0; font-size: 14px;">Medical Report</h2>
                    <div class="patient-info" style="font-size: 11px; margin-top: 30px;">
                        <p style="margin: 2px 0;"><strong>Patient Name:</strong> Detector</p>
                        <p style="margin: 2px 0;"><strong>Date:</strong> ${now.toLocaleDateString()}</p>
                        <p style="margin: 2px 0;"><strong>Time:</strong> ${now.toLocaleTimeString()}</p>
                        <p style="margin: 2px 0;"><strong>Report ID:</strong> RPT-${Math.floor(10000 + Math.random() * 90000)}</p>
                    </div>
                </div>
                
                <div class="image-container" style="flex: 0 0 150px; margin-left: 20px;">
                    <img src="${elements.imagePreview.src}" alt="Analyzed image" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 3px;">
                </div>
            </div>

    
                <div class="results-container" style="margin-top: 10px;">
                    <h3 style="color: #4c97a3; margin: 10px 0 5px 0; font-size: 14px;">${deficiency.name} Deficiency</h3>
                    <div class="severity ${analysis.severity.toLowerCase()}" style="padding: 3px 8px; display: inline-block; border-radius: 3px; font-size: 11px; background-color: ${analysis.severity === 'Severe' ? '#ffcccc' : analysis.severity === 'Moderate' ? '#ffe6cc' : '#e6ffe6'};">
                        ${analysis.severity} severity
                        <span class="confidence">(${analysis.confidence}% confidence)</span>
                    </div>
    
                    <div class="symptoms" style="margin-top: 8px;">
                        <h4 style="color: #4c97a3; margin: 8px 0 3px 0; font-size: 12px;">Symptoms:</h4>
                        <ul style="margin: 0 0 5px 15px; padding-left: 5px; font-size: 11px; line-height: 1.3;">${deficiency.symptoms.map(s => `<li style="margin-bottom: 2px;">${s}</li>`).join('')}</ul>
                    </div>
    
                    <div class="causes" style="margin-top: 8px;">
                        <h4 style="color: #4c97a3; margin: 8px 0 3px 0; font-size: 12px;">Possible Causes:</h4>
                        <ul style="margin: 0 0 5px 15px; padding-left: 5px; font-size: 11px; line-height: 1.3;">${deficiency.causes.map(c => `<li style="margin-bottom: 2px;">${c}</li>`).join('')}</ul>
                    </div>
    
                    <div class="recommendations" style="margin-top: 8px;">
                        <h4 style="color: #4c97a3; margin: 8px 0 3px 0; font-size: 12px;">Recommendations:</h4>
                        <ul style="margin: 0 0 5px 15px; padding-left: 5px; font-size: 11px; line-height: 1.3;">${deficiency.recommendations.map(r => `<li style="margin-bottom: 2px;">${r}</li>`).join('')}</ul>
                    </div>
                </div>
    
                <div class="report-footer" style="margin-top: 15px; font-size: 9px; color: #666; border-top: 1px solid #eee; padding-top: 5px;">
                    <p style="margin: 3px 0;"><strong>Disclaimer:</strong> This report is generated by AI analysis and should be reviewed by a medical professional.</p>
                </div>
            </div>
            `
        };
    }
    function getDeficiencyInfo(type) {
        const deficiencies = {
            A: {
                name: "Vitamin A",
                symptoms: ["Night blindness or difficulty seeing in low light",
                "Dry eyes (xerophthalmia)",
                "Dry, scaly skin",
                "Frequent infections",
                "Delayed wound healing"],
                causes: ["Inadequate dietary intake (lack of liver, fish, dairy, orange vegetables)",
                "Malabsorption disorders (celiac disease, Crohn's disease)",
                "Zinc deficiency (zinc is needed for vitamin A metabolism)",
                "Alcoholism"],
                recommendations: ["Increase consumption of vitamin A-rich foods (sweet potatoes, carrots, spinach, liver)",
                "Consider vitamin A supplement (2500-5000 IU daily) under medical supervision",
                "Treat any underlying malabsorption conditions",
                "Get follow-up blood test in 8-12 weeks",
                "Avoid excessive vitamin A intake which can be toxic"]
            },
            B: {
                name: "Vitamin B",
                symptoms: ["Fatigue and weakness",
                "Pale or yellowish skin",
                "Tingling or numbness in hands/feet",
                "Mouth ulcers",
                "Irritability or depression"],
                causes: ["Poor diet lacking in whole grains, meat, eggs",
                "Pernicious anemia (B12 absorption issue)",
                "Alcoholism",
                "Certain medications (PPIs, metformin)",
                "Autoimmune disorders"],
                recommendations: ["Consume more B-vitamin rich foods (whole grains, eggs, meat, leafy greens)",
                "Consider B-complex supplement",
                "For B12 deficiency, may need sublingual tablets or injections",
                "Address any underlying absorption issues",
                "Limit alcohol consumption"]
            },
            C: {
                name: "Vitamin C",
                symptoms: ["Easy bruising",
                "Slow wound healing",
                "Bleeding gums",
                "Dry, rough, scaly skin",
                "Weak immune system"],
                causes: ["Diet lacking fresh fruits and vegetables",
                "Smoking (increases vitamin C requirements)",
                "Malabsorption disorders",
                "Alcoholism",
                "Extreme dieting"],
                recommendations: ["Increase consumption of citrus fruits, berries, peppers",
                "Consider vitamin C supplement (500-1000 mg daily)",
                "Stop smoking if applicable",
                "Cook vegetables lightly to preserve vitamin C",
                "Monitor symptoms for improvement"]
            },
            D: {
                name: "Vitamin D",
                symptoms: ["Bone pain or tenderness",
                "Muscle weakness",
                "Fatigue and tiredness",
                "Frequent infections",
                "Depression or mood changes"],
                causes: ["Limited sunlight exposure",
                "Darker skin pigmentation",
                "Obesity (vitamin D gets sequestered in fat)",
                "Malabsorption disorders",
                "Strict vegan diet without supplementation"],
                recommendations: ["Get 15-30 minutes of sunlight daily (arms and legs exposed)",
                "Take vitamin D3 supplement (2000-5000 IU daily)",
                "Consume vitamin D-rich foods (fatty fish, fortified dairy)",
                "Have follow-up blood test in 3 months",
                "Consider calcium supplement if levels are also low"]
            },
            E: {
                name: "Vitamin E",
                symptoms: ["Peripheral neuropathy",
                "Muscle weakness",
                "Vision problems",
                "Immune system impairment",
                "Dry, damaged skin"],
                causes: ["Fat malabsorption disorders (cystic fibrosis, liver disease)",
                "Very low-fat diets",
                "Genetic disorders (abetalipoproteinemia)",
                "Premature infants",
                "Long-term parenteral nutrition"],
                recommendations: ["Increase consumption of nuts, seeds, and vegetable oils",
                "Consider vitamin E supplement (100-400 IU daily)",
                "Address any underlying fat absorption issues",
                "Use vitamin E oil for skin issues",
                "Monitor neurological symptoms"]
            }
        };
        
        return deficiencies[type] || deficiencies.A;
    }

    function displayResults(report) {
        try {
            elements.resultsSection.style.display = 'block';
    
            elements.resultsContent.innerHTML = `
            <div id="reportToPrint" class="doctor-report" style="width: 100%;">
                <div style="text-align: center; margin-bottom: 20px;">
                    <h1 style="color: #4c97a3; margin: 0 0 5px 0;">Vitamin Deficiency Detection</h1>
                    <h2 style="color: #333; margin: 0;">Medical Report</h2>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;">
                    <div class="patient-info" style="flex: 1; margin-right: 20px; margin-top: 30px;">
                        <p><strong>Patient Name:</strong> Detector</p>
                        <p><strong>Date:</strong> ${report.date}</p>
                        <p><strong>Time:</strong> ${report.time}</p>
                        <p><strong>Report ID:</strong> ${report.id}</p>
                    </div>
                    
                    <div class="image-container" style="flex: 1; max-width: 300px;">
                        <img src="${report.image}" alt="Analyzed image" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px;">
                    </div>
                </div>
    
                <div style="clear: both;"></div>
    
                <div class="results-container" style="margin-top: 20px;">
                    <h3 style="color: #4c97a3;">${report.deficiency.name} Deficiency</h3>
                    <div class="severity ${report.deficiency.severity.toLowerCase()}" style="padding: 5px 10px; display: inline-block; border-radius: 3px;">
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
    
            <div class="report-actions" style="margin-top: 20px;">
                <button id="downloadPdf" class="download-btn" style="background-color: #4c97a3; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer;">
                    <i class="fas fa-file-pdf"></i> Download PDF Report
                </button>
            </div>`;

            let reportsHistory = [];
        if (localStorage.getItem('reportsHistory')) {
            reportsHistory = JSON.parse(localStorage.getItem('reportsHistory'));
        }
        reportsHistory.push(report);
        localStorage.setItem('reportsHistory', JSON.stringify(reportsHistory));

        // Add event listener for download button
        document.getElementById('downloadPdf').addEventListener('click', () => {
            downloadReport(report);

            });
            
        } catch (error) {
            console.error('Error displaying results:', error);
            alert('Failed to display results');
        }
    }
    
    async function downloadReport(report) {
        const element = document.getElementById('reportToPrint');
        if (!element) return;
    
        try {
            const downloadBtn = document.getElementById('downloadPdf');
            if (downloadBtn) {
                downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating PDF...';
                downloadBtn.disabled = true;
            }
    
            // Create a clone of the element to modify for PDF
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
                    return element.classList && element.classList.contains('report-actions');
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
            const downloadBtn = document.getElementById('downloadPdf');
            if (downloadBtn) {
                downloadBtn.innerHTML = '<i class="fas fa-file-pdf"></i> Download PDF Report';
                downloadBtn.disabled = false;
            }
        }
    }
});