"""
Sistema de Gestión de Temas Dinámicos
Permite cambiar estilos CSS mediante configuración JSON
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from flask import current_app

class ThemeManager:
    """Gestiona los temas dinámicos basados en JSON"""
    
    def __init__(self, themes_dir: str = "static/themes"):
        self.themes_dir = Path(themes_dir)
        self._themes_cache = {}
        self._current_theme = "default"
        
    def get_available_themes(self) -> List[Dict[str, Any]]:
        """Obtiene lista de temas disponibles"""
        themes = []
        
        if not self.themes_dir.exists():
            return themes
            
        for theme_file in self.themes_dir.glob("*.json"):
            try:
                with open(theme_file, 'r', encoding='utf-8') as f:
                    theme_data = json.load(f)
                    themes.append({
                        'id': theme_file.stem,
                        'name': theme_data.get('name', theme_file.stem),
                        'description': theme_data.get('description', ''),
                        'version': theme_data.get('version', '1.0.0')
                    })
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading theme {theme_file}: {e}")
                
        return sorted(themes, key=lambda x: x['name'])
    
    def load_theme(self, theme_id: str) -> Optional[Dict[str, Any]]:
        """Carga un tema específico"""
        if theme_id in self._themes_cache:
            return self._themes_cache[theme_id]
            
        theme_path = self.themes_dir / f"{theme_id}.json"
        
        if not theme_path.exists():
            return None
            
        try:
            with open(theme_path, 'r', encoding='utf-8') as f:
                theme_data = json.load(f)
                self._themes_cache[theme_id] = theme_data
                return theme_data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading theme {theme_id}: {e}")
            return None
    
    def generate_css_variables(self, theme_id: str, mode: str = "light") -> str:
        """Genera variables CSS desde el tema JSON"""
        theme = self.load_theme(theme_id)
        if not theme:
            return ""
            
        css_vars = []
        css_vars.append(f"/* Theme: {theme.get('name', theme_id)} - {mode.title()} Mode */")
        css_vars.append(":root {")
        
        # Colors
        colors = theme.get('colors', {}).get(mode, {})
        for color_name, color_value in colors.items():
            css_vars.append(f"  --{color_name}: {color_value};")
            
        # Typography
        typography = theme.get('typography', {})
        if 'font-sans' in typography:
            css_vars.append(f"  --font-sans: {typography['font-sans']};")
        if 'font-serif' in typography:
            css_vars.append(f"  --font-serif: {typography['font-serif']};")
        if 'font-mono' in typography:
            css_vars.append(f"  --font-mono: {typography['font-mono']};")
            
        # Typography sizes
        sizes = typography.get('sizes', {})
        for size_name, size_value in sizes.items():
            css_vars.append(f"  --font-size-{size_name}: {size_value};")
            
        # Spacing
        spacing = theme.get('spacing', {})
        for spacing_name, spacing_value in spacing.items():
            css_vars.append(f"  --{spacing_name}: {spacing_value};")
            
        # Shadows
        shadows = theme.get('shadows', {})
        for shadow_name, shadow_value in shadows.items():
            css_vars.append(f"  --shadow-{shadow_name}: {shadow_value};")
            
        # Animations
        animations = theme.get('animations', {})
        for anim_name, anim_value in animations.items():
            css_vars.append(f"  --{anim_name}: {anim_value};")
            
        css_vars.append("}")
        
        return "\n".join(css_vars)
    
    def generate_complete_css(self, theme_id: str) -> str:
        """Genera CSS completo con variables y dark mode"""
        theme = self.load_theme(theme_id)
        if not theme:
            return ""
            
        css_parts = []
        
        # CSS base con variables light
        css_parts.append(self.generate_css_variables(theme_id, "light"))
        
        # Dark mode
        css_parts.append("\n/* Dark Mode */")
        css_parts.append(".dark {")
        
        dark_colors = theme.get('colors', {}).get('dark', {})
        for color_name, color_value in dark_colors.items():
            css_parts.append(f"  --{color_name}: {color_value};")
            
        css_parts.append("}")
        
        # Estilos dinámicos basados en tipografía
        css_parts.append(self._generate_typography_css(theme))
        
        return "\n".join(css_parts)
    
    def _generate_typography_css(self, theme: Dict[str, Any]) -> str:
        """Genera CSS para tipografía dinámica"""
        typography = theme.get('typography', {})
        sizes = typography.get('sizes', {})
        
        css = ["\n/* Dynamic Typography */"]
        
        for element in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if element in sizes:
                css.append(f"{element} {{ font-size: var(--font-size-{element}); }}")
                
        if 'body' in sizes:
            css.append(f"body {{ font-size: var(--font-size-body); }}")
            
        if 'lead' in sizes:
            css.append(f".lead {{ font-size: var(--font-size-lead); }}")
            
        return "\n".join(css)
    
    def save_custom_theme(self, theme_id: str, theme_data: Dict[str, Any]) -> bool:
        """Guarda un tema personalizado"""
        try:
            theme_path = self.themes_dir / f"{theme_id}.json"
            
            # Crear directorio si no existe
            self.themes_dir.mkdir(parents=True, exist_ok=True)
            
            with open(theme_path, 'w', encoding='utf-8') as f:
                json.dump(theme_data, f, indent=2, ensure_ascii=False)
                
            # Limpiar cache
            if theme_id in self._themes_cache:
                del self._themes_cache[theme_id]
                
            return True
            
        except (IOError, json.JSONEncodeError) as e:
            print(f"Error saving theme {theme_id}: {e}")
            return False
    
    def delete_theme(self, theme_id: str) -> bool:
        """Elimina un tema personalizado"""
        # No permitir eliminar temas por defecto
        if theme_id in ['default', 'medical-blue', 'warm-clinical']:
            return False
            
        try:
            theme_path = self.themes_dir / f"{theme_id}.json"
            
            if theme_path.exists():
                theme_path.unlink()
                
                # Limpiar cache
                if theme_id in self._themes_cache:
                    del self._themes_cache[theme_id]
                    
                return True
                
        except IOError as e:
            print(f"Error deleting theme {theme_id}: {e}")
            
        return False
    
    def set_current_theme(self, theme_id: str) -> bool:
        """Establece el tema actual"""
        if self.load_theme(theme_id):
            self._current_theme = theme_id
            return True
        return False
    
    def get_current_theme(self) -> str:
        """Obtiene el tema actual"""
        return self._current_theme

# Instancia global
theme_manager = ThemeManager()