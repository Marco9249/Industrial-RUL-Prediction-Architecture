document.addEventListener('DOMContentLoaded', () => {
    let currentStep = 1;
    const totalSteps = 4;
    let activeTab = 'toc';

    // State management for Indices
    const state = {
        toc: [
            { title: "الفصل الأول: المقدمة", level: 1, page: "1" },
            { title: "1.1 المقدمة", level: 2, page: "1" },
            { title: "الفصل الثاني: الإطار النظري والدراسات السابقة", level: 1, page: "10" },
            { title: "الفصل الثالث: المنهجية المتبعة", level: 1, page: "25" },
            { title: "الفصل الرابع: النتائج والمناقشة", level: 1, page: "40" },
            { title: "الفصل الخامس: التوصيات والخاتمة", level: 1, page: "55" }
        ],
        figures: [
            { title: "الشكل (3-1): مسار تدفق البيانات في المشروع", page: "28" },
            { title: "الشكل (4-2): مقارنة استقرار النظام المقترح", page: "45" }
        ],
        tables: [
            { title: "جدول (3-1): معايير ضبط المتحكم FOPID", page: "32" },
            { title: "جدول (4-5): نتائج الأداء الكمية", page: "50" }
        ]
    };

    // Initialize Fields with Found Metadata
    const initData = {
        title_ar: "تحسين منظومات الطاقة الكهروضوئية باستخدام نموذج تنبؤ هجين موجه بالفيزياء",
        title_en: "Improving Photovoltaic Energy Systems using Physics-Guided Hybrid Forecasting Model",
        students: "مازن عبد اللطيف عبد الرحمن حامد\nمحمد الطيب بخيت ادريس\nمحمد خالد عبد الرحمن مبارك\nمحمد عزالدين بابكر عبدالله",
        supervisor: "أ. رفيدة عبدالله إبراهيم محمد"
    };

    Object.keys(initData).forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = initData[id];
    });

    // --- NAVIGATION ---
    const nextBtn = document.getElementById('next-btn');
    const prevBtn = document.getElementById('prev-btn');
    const navLinks = document.querySelectorAll('.nav-links li');
    const steps = document.querySelectorAll('.step-content');

    function showStep(step) {
        steps.forEach(s => s.classList.remove('active'));
        navLinks.forEach(l => l.classList.remove('active'));
        
        document.getElementById(`step-${step}`).classList.add('active');
        document.querySelector(`.nav-links li[data-step="${step}"]`).classList.add('active');
        document.getElementById('current-step-label').innerText = step;

        prevBtn.disabled = step === 1;
        nextBtn.innerText = step === totalSteps ? 'معاينة نهائية' : 'التالي (Step '+(step+1)+')';
        
        if(step === 4) renderMasterPreview();
    }

    nextBtn.addEventListener('click', () => { if (currentStep < totalSteps) { currentStep++; showStep(currentStep); } });
    prevBtn.addEventListener('click', () => { if (currentStep > 1) { currentStep--; showStep(currentStep); } });

    // --- TABS & TABLE EDITING ---
    window.switchTab = (tab) => {
        activeTab = tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelector(`.tab-btn[onclick="switchTab('${tab}')"]`).classList.add('active');
        renderEditTable();
    };

    function renderEditTable() {
        const tbody = document.querySelector('#edit-toc-table tbody');
        tbody.innerHTML = '';
        state[activeTab].forEach((item, index) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td contenteditable="true" onblur="updateState('${activeTab}', ${index}, 'title', this.innerText)">${item.title}</td>
                ${activeTab === 'toc' ? `<td><select onchange="updateState('toc', ${index}, 'level', this.value)">
                    <option value="1" ${item.level==1?'selected':''}>فصل الرئيسي</option>
                    <option value="2" ${item.level==2?'selected':''}>عنوان فرعي</option>
                </select></td>` : '<td>-</td>'}
                <td contenteditable="true" onblur="updateState('${activeTab}', ${index}, 'page', this.innerText)">${item.page}</td>
                <td><button onclick="removeItem('${activeTab}', ${index})" style="background:none; border:none; cursor:pointer;">🗑️</button></td>
            `;
            tbody.appendChild(tr);
        });
    }

    window.updateState = (list, idx, key, val) => { state[list][idx][key] = val; };
    window.removeItem = (list, idx) => { state[list].splice(idx, 1); renderEditTable(); };
    window.addItem = () => {
        state[activeTab].push({ title: 'بند جديد', level: 1, page: '0' });
        renderEditTable();
    };

    // --- PREVIEW RENDERING ---
    function renderMasterPreview() {
        const preview = document.getElementById('page-preview');
        const d = {
            univ: document.getElementById('univ_name').value,
            college: document.getElementById('college_name').value,
            dept: document.getElementById('dept_name').value,
            titleAr: document.getElementById('title_ar').value,
            titleEn: document.getElementById('title_en').value,
            students: document.getElementById('students_list').value.split('\n'),
            supervisor: document.getElementById('supervisor_name').value
        };

        let html = `
            <!-- TITLE PAGE -->
            <div class="preview-page">
                <div style="text-align: right; font-weight: bold; margin-bottom: 50px;">
                    <p>${d.univ}</p><p>${d.college}</p><p>${d.dept}</p>
                </div>
                <div style="margin: 100px 0; text-align: center;">
                    <h1 style="font-size: 24pt; margin-bottom: 20px;">${d.titleAr}</h1>
                    <h2 style="font-size: 18pt; font-weight: normal;">${d.titleEn}</h2>
                </div>
                <div style="margin-top: 50px; text-align: right; width: 100%;">
                    <p style="font-weight: bold; margin-bottom: 10px;">إعداد:</p>
                    ${d.students.map(s => `<p>${s}</p>`).join('')}
                    <p style="font-weight: bold; margin-top: 30px; margin-bottom: 10px;">إشراف:</p>
                    <p>${d.supervisor}</p>
                </div>
                <div style="position: absolute; bottom: 80px; left: 0; right: 0; text-align: center;">
                    <p style="font-weight: bold;">2026م</p>
                </div>
            </div>

            <div style="page-break-before: always;"></div>

            <!-- TABLE OF CONTENTS (REAL TABLE) -->
            <div class="preview-page">
                <h2 style="text-align: center; margin-bottom: 30px;">فهرس المحتويات</h2>
                <table class="academic-table">
                    <thead><tr><th>الموضوع</th><th>رقم الصفحة</th></tr></thead>
                    <tbody>
                        ${state.toc.map(item => `
                            <tr>
                                <td style="padding-right: ${item.level*15}pt; ${item.level==1?'font-weight:bold;':''}">${item.title}</td>
                                <td style="text-align: center;">${item.page}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>

            <div style="page-break-before: always;"></div>

            <!-- LIST OF FIGURES (REAL TABLE) -->
            <div class="preview-page">
                <h2 style="text-align: center; margin-bottom: 30px;">فهرس الأشكال</h2>
                <table class="academic-table">
                    <thead><tr><th>عنوان الشكل</th><th>رقم الصفحة</th></tr></thead>
                    <tbody>
                        ${state.figures.map(item => `
                            <tr><td>${item.title}</td><td style="text-align: center;">${item.page}</td></tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
        preview.innerHTML = html;
    }

    // --- EXPORTS ---
    window.exportPDF = () => {
        const element = document.getElementById('page-preview');
        const opt = {
            margin:       0,
            filename:     'Research_Prelims.pdf',
            image:        { type: 'jpeg', quality: 0.98 },
            html2canvas:  { scale: 2, useCORS: true },
            jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
        };
        html2pdf().set(opt).from(element).save();
    };

    window.exportWord = () => {
        // Docx.js Core Logic (Summary)
        alert("جاري تجهيز ملف Word... سيتم تنزيله فوراً.");
        const doc = new docx.Document({
            sections: [{
                properties: {},
                children: [
                    new docx.Paragraph({ text: "فهرس المحتويات", heading: docx.HeadingLevel.HEADING_1, alignment: docx.AlignmentType.CENTER }),
                    new docx.Table({
                        rows: state.toc.map(item => new docx.TableRow({
                            children: [
                                new docx.TableCell({ children: [new docx.Paragraph({ text: item.title })] }),
                                new docx.TableCell({ children: [new docx.Paragraph({ text: item.page })] }),
                            ],
                        })),
                    }),
                ],
            }],
        });

        docx.Packer.toBlob(doc).then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "Research_Prelims.docx";
            a.click();
        });
    };

    renderEditTable();
});
