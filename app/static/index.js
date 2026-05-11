document.addEventListener('DOMContentLoaded', () => {
    // Tenure Slider
    const tenureSlider = document.getElementById('tenure-slider');
    const tenureVal = document.getElementById('tenure-val');

    tenureSlider.addEventListener('input', (e) => {
        tenureVal.textContent = `${e.target.value} Mo`;
    });

    // Toggle Buttons (Gender)
    const toggleBtns = document.querySelectorAll('.toggle-btn');
    toggleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const parent = btn.parentElement;
            parent.querySelector('.active').classList.remove('active');
            btn.classList.add('active');
        });
    });

    // Service Buttons
    const serviceBtns = document.querySelectorAll('.service-btn');
    serviceBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            btn.classList.toggle('active');
        });
    });

    // Predict Button
    const predictBtn = document.getElementById('predict-btn');
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultDisplay = document.getElementById('result-display');

    predictBtn.addEventListener('click', async () => {
        predictBtn.textContent = 'PROCESSING...';
        predictBtn.disabled = true;

        try {
            // Collect form data (though not used in mock, good practice)
            const formData = {
                gender: document.querySelector('#gender-toggle .active').dataset.value,
                senior: document.getElementById('senior-citizen').checked,
                partner: document.getElementById('partner-status').checked,
                dependents: document.getElementById('dependents').checked,
                internet: document.getElementById('internet-service').value,
                phone: document.getElementById('phone-service').value,
                tenure: tenureSlider.value,
                contract: document.getElementById('contract-type').value,
                payment: document.getElementById('payment-method').value,
                monthly: document.getElementById('monthly-charges').value,
                total: document.getElementById('total-charges').value
            };

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const data = await response.json();
            updateResultUI(data);

        } catch (error) {
            console.error('Prediction error:', error);
            alert('Failed to generate prediction. Please try again.');
        } finally {
            predictBtn.textContent = 'GENERATE PREDICTION';
            predictBtn.disabled = false;
        }
    });

    function updateResultUI(data) {
        resultPlaceholder.style.display = 'none';
        resultDisplay.style.display = 'flex';

        const churnStatus = document.getElementById('churn-status');
        churnStatus.textContent = data.status;
        churnStatus.style.color = data.status === 'CHURN LIKELY' ? '#ef4444' : '#2ecc71';

        // Update Donut
        const probVal = document.getElementById('prob-val');
        const donutSegment = document.getElementById('donut-segment');
        probVal.textContent = `${data.probability}%`;
        
        // Calculate dasharray (circumference is 2 * PI * 40 approx 251.2)
        // Using stroke-dasharray: percent 100 for simplicity if using percent based dash
        donutSegment.style.strokeDasharray = `${data.probability} 100`;
        donutSegment.style.stroke = data.probability > 70 ? '#ef4444' : (data.probability > 30 ? '#f59e0b' : '#2ecc71');

        // Update Drivers
        const driversContainer = document.getElementById('drivers-container');
        // Remove existing driver items except the header
        const items = driversContainer.querySelectorAll('.driver-item');
        items.forEach(item => item.remove());

        data.drivers.forEach(driver => {
            const div = document.createElement('div');
            div.className = 'driver-item';
            
            // Randomize impact color slightly for visual variety
            const color = driver.impact > 0 ? '#ef4444' : '#2ecc71';
            const impactLabel = driver.impact > 0 ? `+${driver.impact}%` : `${driver.impact}%`;

            div.innerHTML = `
                <div class="driver-info">
                    <span>${driver.feature}</span>
                    <span style="color: ${color}">${impactLabel}</span>
                </div>
                <div class="progress-bg">
                    <div class="progress-fill" style="width: ${Math.abs(driver.impact) * 2}%; background: ${color}"></div>
                </div>
            `;
            driversContainer.appendChild(div);
        });

        // Update Insight
        document.getElementById('insight-text').textContent = data.insight;

        // Smooth scroll to results on mobile
        if (window.innerWidth <= 1024) {
            resultDisplay.scrollIntoView({ behavior: 'smooth' });
        }
    }
});
