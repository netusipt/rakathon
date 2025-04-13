document.addEventListener('DOMContentLoaded', function() {
    // Enable/disable micrometastases fields based on positive nodes value
    const positiveNodesInput = document.getElementById('positiveNodesInput');
    const micrometastasesRadios = document.querySelectorAll('input[name="micrometastases_only"]');
    
    if (positiveNodesInput) {
        positiveNodesInput.addEventListener('input', function() {
            const positiveNodesValue = parseInt(this.value);
            
            // Enable micrometastases options only if positive nodes is 1
            const enableMicrometastases = positiveNodesValue === 1;
            
            micrometastasesRadios.forEach(radio => {
                radio.disabled = !enableMicrometastases;
                if (!enableMicrometastases) {
                    radio.checked = false;
                }
            });
            
            // If positive nodes is 1, make micrometastases selection required
            if (enableMicrometastases) {
                micrometastasesRadios[0].required = true;
            } else {
                micrometastasesRadios[0].required = false;
            }
        });
    }
    
    // Form validation
    const form = document.getElementById('riskForm');
    if (form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                
                // Create alert for missing fields
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-danger mt-3';
                alertDiv.role = 'alert';
                alertDiv.textContent = 'Vyplňte prosím všechna povinná pole.';
                
                // Check if alert is already displayed
                const existingAlert = form.querySelector('.alert-danger');
                if (!existingAlert) {
                    form.prepend(alertDiv);
                }
                
                // Scroll to the top of the form
                form.scrollIntoView({ behavior: 'smooth' });
            }
            
            form.classList.add('was-validated');
        });
    }
}); 