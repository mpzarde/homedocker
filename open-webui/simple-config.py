#!/usr/bin/env python3
"""
Simple Ollama Router Configuration Manager
(Uses only built-in Python modules)

Usage:
    python3 simple-config.py list                    # Show current configuration  
    python3 simple-config.py set dev "new-model:7b"  # Change dev model
    python3 simple-config.py set-url "http://new-server:11434"  # Change Ollama URL
"""

import yaml
import sys
from pathlib import Path

CONFIG_FILE = Path("ollama-router/config.yaml")

def load_config():
    """Load the current configuration"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def save_config(config):
    """Save configuration back to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print("✅ Configuration saved successfully!")
        print("\n🔄 To apply changes, run:")
        print("   docker-compose up -d --build ollama-router")
    except Exception as e:
        print(f"Error saving config: {e}")
        sys.exit(1)

def list_config():
    """Display current configuration"""
    config = load_config()
    
    print("🤖 Ollama Router Configuration")
    print("=" * 50)
    
    print("\n📝 Model Assignments:")
    for intent, model in config['models'].items():
        emoji = {"dev": "💻", "writing": "✍️", "general": "🧠", "classifier": "🔍"}.get(intent, "🤖")
        print(f"  {emoji} {intent:<12}: {model}")
    
    print(f"\n🔗 Ollama Backend:")
    print(f"  {config['ollama']['base_url']}")
    
    print(f"\n⚙️  Router Settings:")
    for key, value in config['router'].items():
        print(f"  {key:<25}: {value}")
    
    print(f"\n🔍 Keywords (showing first 3 per category):")
    for category, keywords in config['keywords'].items():
        emoji = {"dev": "💻", "writing": "✍️", "new_conversation": "🔄"}.get(category, "🔍")
        print(f"  {emoji} {category}:")
        for kw in keywords[:3]:
            print(f"    - {kw}")
        if len(keywords) > 3:
            print(f"    ... and {len(keywords) - 3} more")
        print()

def set_model(intent, model):
    """Set model for a specific intent"""
    config = load_config()
    
    if intent not in config['models']:
        print(f"❌ Error: Unknown intent '{intent}'")
        print(f"📋 Valid intents: {list(config['models'].keys())}")
        sys.exit(1)
    
    old_model = config['models'][intent]
    config['models'][intent] = model
    
    emoji = {"dev": "💻", "writing": "✍️", "general": "🧠", "classifier": "🔍"}.get(intent, "🤖")
    print(f"{emoji} Changed {intent} model:")
    print(f"  From: {old_model}")
    print(f"  To:   {model}")
    
    save_config(config)

def set_ollama_url(url):
    """Change Ollama backend URL"""
    config = load_config()
    
    old_url = config['ollama']['base_url']
    config['ollama']['base_url'] = url
    
    print(f"🔗 Changed Ollama backend URL:")
    print(f"  From: {old_url}")
    print(f"  To:   {url}")
    
    save_config(config)

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        list_config()
    elif command == "set" and len(sys.argv) == 4:
        set_model(sys.argv[2], sys.argv[3])
    elif command == "set-url" and len(sys.argv) == 3:
        set_ollama_url(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()
