document.addEventListener("DOMContentLoaded", function() {
    // ============== DOM ELEMENTS ==============
    const navbar = document.querySelector('.navbar');
    const hero = document.querySelector('.hero');
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    const errorPopup = document.getElementById('error-popup');
    const closeBtn = document.querySelector('.close-btn');
    const loginBtn = document.querySelector('.login-btn');

    // ============== SCROLL AND ANIMATION VARIABLES ==============
    let lastScroll = 0;
    let featureBoxesAnimated = false;
    const featureBoxes = document.querySelectorAll('.feature-box');
    const featuresSection = document.querySelector('.features');
    const fadeElements = document.querySelectorAll('.fade-in');
    const slideElements = document.querySelectorAll('.slide-up');

    // ============== INITIAL SETUP ==============
    function initializePage() {
        // Navbar setup
        if (navbar) {
            const navbarHeight = navbar.offsetHeight;
            navbar.classList.add('visible');
            if (hero) hero.style.marginTop = `${navbarHeight}px`;
        }

        // Initialize feature boxes
        if (featureBoxes.length > 0) {
            featureBoxes.forEach(box => {
                box.style.opacity = '0';
                box.style.transform = 'translateY(30px)';
                box.style.transition = 'all 0.6s cubic-bezier(0.25, 0.1, 0.25, 1)';
            });
        }

        // Setup forms
        addExampleUser();
        if (loginForm) setupLogin();
        if (registerForm) setupRegistration();

        // Initial animation check
        animateOnScroll();
    }

    // ============== SCROLL EVENT HANDLER ==============
    function handleScroll() {
        const currentScroll = window.pageYOffset;
        
        // Navbar Background Effect
        if (navbar) {
            if (currentScroll > 50) {
                navbar.classList.add('scrolled');
                if (hero) hero.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
                if (hero) hero.classList.remove('scrolled');
            }
        }

        // Feature Box Animation
        if (featuresSection) {
            if (currentScroll > lastScroll) {
                const featuresTop = featuresSection.offsetTop;
                const featuresHeight = featuresSection.offsetHeight;
                const scrollPosition = currentScroll + window.innerHeight;
                
                if (scrollPosition > featuresTop + 100 && 
                    currentScroll < featuresTop + featuresHeight - 100) {
                    
                    if (!featureBoxesAnimated) {
                        animateFeatureBoxes();
                        featureBoxesAnimated = true;
                    }
                }
            } 
            else if (currentScroll < featuresSection.offsetTop - window.innerHeight) {
                featureBoxesAnimated = false;
                resetFeatureBoxes();
            }
        }

        lastScroll = currentScroll;
        animateOnScroll();
    }

    // ============== ANIMATION FUNCTIONS ==============
    function animateFeatureBoxes() {
        featureBoxes.forEach((box, index) => {
            setTimeout(() => {
                box.style.opacity = '1';
                box.style.transform = 'translateY(0)';
                
                setTimeout(() => {
                    box.addEventListener('mouseenter', () => {
                        box.style.transform = 'translateY(-10px) scale(1.05)';
                        box.style.boxShadow = '0 15px 30px rgba(0,0,0,0.3)';
                    });
                    
                    box.addEventListener('mouseleave', () => {
                        box.style.transform = 'translateY(0) scale(1)';
                        box.style.boxShadow = '0 5px 15px rgba(0,0,0,0.2)';
                    });
                }, 600 + (index * 100));
            }, index * 200);
        });
    }

    function resetFeatureBoxes() {
        featureBoxes.forEach(box => {
            box.style.opacity = '0';
            box.style.transform = 'translateY(30px)';
            box.style.boxShadow = 'none';
        });
    }

    function animateOnScroll() {
        const windowHeight = window.innerHeight;
        const triggerPoint = 100;
        
        fadeElements.forEach(el => {
            const elementPos = el.getBoundingClientRect().top;
            if (elementPos < windowHeight - triggerPoint) {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }
        });
        
        slideElements.forEach(el => {
            const elementPos = el.getBoundingClientRect().top;
            if (elementPos < windowHeight - triggerPoint) {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }
        });
    }

    // ============== AUTHENTICATION FUNCTIONS ==============
    function setupLogin() {
        try {
            loginForm.addEventListener('submit', handleLogin);
            
            if (closeBtn) {
                closeBtn.addEventListener('click', hideError);
            }
            
            if (errorPopup) {
                errorPopup.addEventListener('click', function(e) {
                    if (e.target === errorPopup) hideError();
                });
            }
        } catch (error) {
            console.error('Login setup failed:', error);
        }
    }

    async function handleLogin(e) {
        e.preventDefault();
        const email = emailInput.value.trim();
        const password = passwordInput.value;

        // Validation
        if (!validateEmail(email)) {
            showError("Please enter a valid email address");
            return;
        }
        if (password.length < 8) {
            showError("Password must be at least 8 characters");
            return;
        }

        loginBtn.disabled = true;
        loginBtn.textContent = "Authenticating...";

        try {
            await new Promise(resolve => setTimeout(resolve, 1500));
            const user = authenticateUser(email, password);
            
            if (user) {
                showSuccess("Login successful! Redirecting...", () => {
                    window.location.href = "dashboard.html";
                });
            } else {
                showError("Invalid email or password");
            }
        } catch (error) {
            console.error("Login error:", error);
            showError("Login failed. Please try again later.");
        } finally {
            loginBtn.disabled = false;
            loginBtn.textContent = "Login";
        }
    }

    // ============== REGISTRATION FUNCTIONS ==============
    function setupRegistration() {
        try {
            const passwordInput = document.getElementById('password');
            const strengthMeter = document.querySelector('.strength-meter');
            
            if (passwordInput && strengthMeter) {
                passwordInput.addEventListener('input', function() {
                    updatePasswordStrength(passwordInput.value, strengthMeter);
                });
            }

            registerForm.addEventListener('submit', handleRegister);
        } catch (error) {
            console.error('Registration setup failed:', error);
        }
    }

    async function handleRegister(e) {
        e.preventDefault();
        const registerBtn = document.querySelector('.login-btn');
        const name = document.getElementById('name').value.trim();
        const email = document.getElementById('email').value.trim();
        const phone = document.getElementById('phone').value.trim();
        const age = document.getElementById('age').value;
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirm-password').value;

        // Validation
        if (!validateName(name)) {
            showError("Please enter a valid full name (min 3 characters)");
            return;
        }
        if (!validateEmail(email)) {
            showError("Please enter a valid email address");
            return;
        }
        if (!validatePhone(phone)) {
            showError("Please enter a valid phone number (10 digits)");
            return;
        }
        if (!validateAge(age)) {
            showError("Please enter a valid age (1-120)");
            return;
        }
        if (password.length < 8) {
            showError("Password must be at least 8 characters");
            return;
        }
        if (password !== confirmPassword) {
            showError("Passwords do not match");
            return;
        }

        registerBtn.disabled = true;
        registerBtn.textContent = "Registering...";

        try {
            if (userExists(email)) {
                showError("Email already registered. Please login instead.");
                return;
            }

            const user = {
                id: generateId(),
                name,
                email,
                phone,
                age: parseInt(age),
                password,
                createdAt: new Date().toISOString()
            };

            saveUser(user);
            showSuccess("Registration successful! Redirecting to login...", () => {
                window.location.href = "login.html";
            });
        } catch (error) {
            console.error("Registration error:", error);
            showError("Registration failed. Please try again.");
        } finally {
            registerBtn.disabled = false;
            registerBtn.textContent = "Register";
        }
    }

    // ============== DATABASE FUNCTIONS ==============
    function authenticateUser(email, password) {
        const users = getUsers();
        return users.find(user => user.email === email && user.password === password);
    }

    function userExists(email) {
        const users = getUsers();
        return users.some(user => user.email === email);
    }

    function saveUser(user) {
        const users = getUsers();
        users.push(user);
        localStorage.setItem('users', JSON.stringify(users));
    }

    function getUsers() {
        return JSON.parse(localStorage.getItem('users') || '[]');
    }

    function generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    // ============== UTILITY FUNCTIONS ==============
    function updatePasswordStrength(password, meter) {
        const strength = calculatePasswordStrength(password);
        meter.style.width = `${(strength + 1) * 20}%`;
        meter.style.backgroundColor = [
            '#ff0000', '#ff5a00', '#ffbb00', '#a4ff00', '#00ff00'
        ][strength];
    }

    function calculatePasswordStrength(password) {
        let strength = 0;
        if (password.length > 10) strength++;
        if (password.length > 15) strength++;
        if (/[A-Z]/.test(password)) strength++;
        if (/[0-9]/.test(password)) strength++;
        if (/[^A-Za-z0-9]/.test(password)) strength++;
        return Math.min(strength, 4);
    }

    function addExampleUser() {
        const users = getUsers();
        const sampleEmail = "detector@gmail.com";
        const alreadyExists = users.some(u => u.email === sampleEmail);
        if (!alreadyExists) {
            const sampleUser = {
                id: generateId(),
                name: "Detector",
                email: sampleEmail,
                phone: "1234567890",
                age: 22,
                password: "Detector@1234",
                createdAt: new Date().toISOString()
            };
            users.push(sampleUser);
            localStorage.setItem('users', JSON.stringify(users));
        }
    }

    function validateName(name) {
        return name.length >= 3;
    }

    function validateEmail(email) {
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    }

    function validatePhone(phone) {
        const digits = phone.replace(/\D/g, '');
        return digits.length === 10 ;
    }

    function validateAge(age) {
        const num = parseInt(age);
        return !isNaN(num) && num >= 1 && num <= 120;
    }

    // ============== POPUP FUNCTIONS ==============
    function showError(message) {
        if (!errorPopup) {
            alert(`Error: ${message}`);
            return;
        }
        
        const errorMessage = errorPopup.querySelector('p');
        if (errorMessage) {
            errorMessage.textContent = message;
        }
        
        errorPopup.style.display = "flex";
        setTimeout(hideError, 5000);
    }

    function hideError() {
        if (errorPopup) {
            errorPopup.style.display = "none";
        }
    }

    function showSuccess(message, callback) {
        const successPopup = document.createElement('div');
        successPopup.className = 'success-popup';
        successPopup.innerHTML = `
            <div class="success-content">
                <p>${message}</p>
            </div>
        `;
        document.body.appendChild(successPopup);
        
        setTimeout(() => {
            successPopup.remove();
            if (callback) callback();
        }, 1500);
    }

    // ============== GLOBAL FUNCTIONS ==============
    window.allowOnlyNumbers = function(e) {
        const key = e.key;
        return /[0-9+\-() ]|Backspace|Delete|ArrowLeft|ArrowRight|Tab/.test(key);
    };

    window.formatPhoneNumber = function(input) {
        // Remove all non-numeric characters
        let numbers = input.value.replace(/\D/g, '');
        
        // Limit to 15 digits (or 10 if needed)
        input.value = numbers.slice(0, 10);
    lue = numbers;
    };


    

    // ============== EVENT LISTENERS ==============
    window.addEventListener('scroll', handleScroll);

    // ============== INITIALIZE PAGE ==============
    initializePage();
});