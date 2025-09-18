/**
 * Sistema de Temas Dinámicos
 * Gestiona el cambio automático de temas mediante JSON y CSS
 */

class ThemeSystem {
    constructor() {
        this.currentTheme = localStorage.getItem('selectedTheme') || 'default';
        this.darkMode = localStorage.getItem('darkMode') === 'true';
        this.themes = [];
        
        this.init();
    }
    
    async init() {
        await this.loadAvailableThemes();
        this.setupThemeSelector();
        this.applyCurrentTheme();
        this.setupEventListeners();
    }
    
    async loadAvailableThemes() {
        try {
            const response = await fetch('/api/themes');
            const data = await response.json();
            
            if (data.success) {
                this.themes = data.themes;
                this.currentTheme = data.current || this.currentTheme;
            }
        } catch (error) {
            console.error('Error loading themes:', error);
        }
    }
    
    setupThemeSelector() {
        // Crear selector de temas si no existe
        if (!document.getElementById('themeSelector')) {
            this.createThemeSelector();
        }
        
        this.populateThemeSelector();
    }
    
    createThemeSelector() {
        const selector = document.createElement('div');
        selector.id = 'themeSelector';
        selector.className = 'theme-selector';
        selector.innerHTML = `
            <button class="theme-toggle-btn" id="themeToggleBtn" title="Selector de Temas">
                <i class="fas fa-palette"></i>
            </button>
            <div class="theme-dropdown" id="themeDropdown">
                <div class="theme-dropdown-header">
                    <h6><i class="fas fa-palette me-2"></i>Seleccionar Tema</h6>
                </div>
                <div class="theme-list" id="themeList">
                    <!-- Themes will be populated here -->
                </div>
                <div class="theme-dropdown-footer">
                    <button class="btn btn-sm btn-outline-primary" id="customThemeBtn">
                        <i class="fas fa-plus me-1"></i>Crear Personalizado
                    </button>
                </div>
            </div>
        `;
        
        // Insertar después del dark mode toggle
        const darkModeToggle = document.getElementById('darkModeToggle');
        if (darkModeToggle) {
            darkModeToggle.parentNode.insertBefore(selector, darkModeToggle.nextSibling);
        } else {
            document.body.appendChild(selector);
        }
    }
    
    populateThemeSelector() {
        const themeList = document.getElementById('themeList');
        if (!themeList) return;
        
        themeList.innerHTML = '';
        
        this.themes.forEach(theme => {
            const themeItem = document.createElement('div');
            themeItem.className = `theme-item ${theme.id === this.currentTheme ? 'active' : ''}`;
            themeItem.innerHTML = `
                <div class="theme-preview" data-theme-id="${theme.id}">
                    <div class="theme-colors">
                        <div class="color-dot primary"></div>
                        <div class="color-dot accent"></div>
                        <div class="color-dot secondary"></div>
                    </div>
                    <div class="theme-info">
                        <div class="theme-name">${theme.name}</div>
                        <div class="theme-description">${theme.description}</div>
                    </div>
                    ${theme.id === this.currentTheme ? '<i class="fas fa-check theme-check"></i>' : ''}
                </div>
            `;
            
            themeList.appendChild(themeItem);
        });
    }
    
    setupEventListeners() {
        // Toggle del selector de temas
        const toggleBtn = document.getElementById('themeToggleBtn');
        const dropdown = document.getElementById('themeDropdown');
        
        if (toggleBtn && dropdown) {
            toggleBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                dropdown.classList.toggle('show');
            });
            
            // Cerrar al hacer click fuera
            document.addEventListener('click', (e) => {
                if (!e.target.closest('.theme-selector')) {
                    dropdown.classList.remove('show');
                }
            });
        }
        
        // Selección de temas
        const themeList = document.getElementById('themeList');
        if (themeList) {
            themeList.addEventListener('click', (e) => {
                const themePreview = e.target.closest('.theme-preview');
                if (themePreview) {
                    const themeId = themePreview.dataset.themeId;
                    this.applyTheme(themeId);
                }
            });
        }
        
        // Botón de tema personalizado
        const customBtn = document.getElementById('customThemeBtn');
        if (customBtn) {
            customBtn.addEventListener('click', () => {
                this.openCustomThemeModal();
            });
        }
    }
    
    async applyTheme(themeId) {
        try {
            // Establecer tema en el servidor
            const response = await fetch(`/api/themes/current/${themeId}`, {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentTheme = themeId;
                localStorage.setItem('selectedTheme', themeId);
                
                // Aplicar CSS del tema
                await this.loadThemeCSS(themeId);
                
                // Actualizar UI
                this.populateThemeSelector();
                this.showNotification(`Tema "${data.current}" aplicado correctamente`, 'success');
                
                // Cerrar dropdown
                document.getElementById('themeDropdown')?.classList.remove('show');
            } else {
                this.showNotification('Error al aplicar tema', 'error');
            }
        } catch (error) {
            console.error('Error applying theme:', error);
            this.showNotification('Error de conexión al cambiar tema', 'error');
        }
    }
    
    async loadThemeCSS(themeId) {
        try {
            // Remover CSS anterior del tema
            const existingThemeCSS = document.getElementById('dynamicThemeCSS');
            if (existingThemeCSS) {
                existingThemeCSS.remove();
            }
            
            // Cargar nuevo CSS
            const response = await fetch(`/api/themes/${themeId}/css`);
            const cssContent = await response.text();
            
            // Crear y aplicar nuevo CSS
            const style = document.createElement('style');
            style.id = 'dynamicThemeCSS';
            style.textContent = cssContent;
            document.head.appendChild(style);
            
        } catch (error) {
            console.error('Error loading theme CSS:', error);
        }
    }
    
    async applyCurrentTheme() {
        await this.loadThemeCSS(this.currentTheme);
        
        // Aplicar dark mode si está activo
        if (this.darkMode) {
            document.body.classList.add('dark');
        }
    }
    
    openCustomThemeModal() {
        // Por ahora, mostrar mensaje de funcionalidad futura
        this.showNotification('Editor de temas personalizado próximamente', 'info');
    }
    
    showNotification(message, type = 'info') {
        // Crear notificación
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed theme-notification`;
        notification.style.cssText = 'top: 80px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-dismiss después de 3 segundos
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// CSS para el selector de temas
const themeSelectorCSS = `
.theme-selector {
    position: fixed;
    top: 1rem;
    right: 5rem;
    z-index: 1000;
}

.theme-toggle-btn {
    background-color: var(--card);
    border: 2px solid var(--border);
    border-radius: 50%;
    width: 3rem;
    height: 3rem;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: var(--shadow-md);
    color: var(--foreground);
}

.theme-toggle-btn:hover {
    background-color: var(--primary);
    color: var(--primary-foreground);
    transform: scale(1.1);
}

.theme-dropdown {
    position: absolute;
    top: calc(100% + 0.5rem);
    right: 0;
    background-color: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-xl);
    min-width: 320px;
    max-height: 400px;
    overflow-y: auto;
    opacity: 0;
    visibility: hidden;
    transform: translateY(-10px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.theme-dropdown.show {
    opacity: 1;
    visibility: visible;
    transform: translateY(0);
}

.theme-dropdown-header {
    padding: 1rem;
    border-bottom: 1px solid var(--border);
    background-color: var(--muted);
}

.theme-dropdown-header h6 {
    margin: 0;
    color: var(--foreground);
    font-weight: 600;
}

.theme-list {
    padding: 0.5rem;
}

.theme-item {
    margin-bottom: 0.25rem;
}

.theme-preview {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem;
    border-radius: calc(var(--radius) - 2px);
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
}

.theme-preview:hover {
    background-color: var(--accent);
    color: var(--accent-foreground);
}

.theme-item.active .theme-preview {
    background-color: var(--primary);
    color: var(--primary-foreground);
}

.theme-colors {
    display: flex;
    gap: 0.25rem;
}

.color-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 1px solid rgba(0,0,0,0.1);
}

.color-dot.primary {
    background-color: var(--primary);
}

.color-dot.accent {
    background-color: var(--accent);
}

.color-dot.secondary {
    background-color: var(--secondary);
}

.theme-info {
    flex: 1;
}

.theme-name {
    font-weight: 500;
    font-size: 0.875rem;
    line-height: 1.2;
}

.theme-description {
    font-size: 0.75rem;
    opacity: 0.8;
    line-height: 1.3;
}

.theme-check {
    position: absolute;
    right: 0.75rem;
    top: 50%;
    transform: translateY(-50%);
    color: currentColor;
}

.theme-dropdown-footer {
    padding: 0.75rem 1rem;
    border-top: 1px solid var(--border);
    background-color: var(--muted);
}

.theme-notification {
    animation: slideInFromRight 0.3s ease-out;
}

@keyframes slideInFromRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}
`;

// Insertar CSS del selector de temas
const style = document.createElement('style');
style.textContent = themeSelectorCSS;
document.head.appendChild(style);

// Inicializar sistema de temas cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    window.themeSystem = new ThemeSystem();
});