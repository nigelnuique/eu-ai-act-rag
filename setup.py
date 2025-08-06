#!/usr/bin/env python3
"""
Setup script for EU AI Act RAG System
This script handles dependency installation and database creation.
"""

import os
import subprocess
import sys
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"🚀 {title}")
    print("=" * 70)

def print_step(step_num, description):
    """Print a formatted step."""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 50)

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install dependencies with error handling."""
    print("📦 Installing dependencies...")
    
    # Install packages one by one to handle conflicts better
    packages = [
        "rank-bm25",
        "streamlit==1.39.0",
        "langchain==0.3.7", 
        "langchain-openai==0.1.6",
        "langchain-community",
        "openai==1.57.0",
        "tiktoken==0.8.0",
        "beautifulsoup4==4.12.3",
        "requests==2.32.3",
        "python-dotenv==1.0.0"
    ]
    
    failed_packages = []
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                package, "--no-deps" if package == "rank-bm25" else ""
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {package}, will try alternative")
            failed_packages.append(package)
    
    # Try to install failed packages without version constraints
    for package in failed_packages:
        try:
            base_package = package.split("==")[0]
            print(f"Trying to install {base_package} without version constraint...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", base_package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {base_package} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {base_package}")
    
    # Check if critical packages are available
    critical_packages = ["langchain", "openai", "streamlit", "rank_bm25"]
    missing = []
    
    for package in critical_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is available")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} is missing")
    
    if missing:
        print(f"⚠️ Some packages are missing: {missing}")
        print("You may need to install them manually:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        return len(missing) == 0
    
    print("✅ All critical dependencies are available")
    return True

def check_api_key():
    """Check if OpenAI API key is available."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and api_key.startswith('sk-'):
            print("✅ OpenAI API key found in environment")
            return api_key
        
        print("⚠️ OpenAI API key not found in .env file")
        api_key = input("Please enter your OpenAI API key: ").strip()
        
        if api_key and api_key.startswith('sk-'):
            # Save to .env file
            with open('.env', 'a') as f:
                f.write(f"\nOPENAI_API_KEY={api_key}\n")
            print("✅ API key saved to .env file")
            return api_key
        else:
            print("❌ Invalid API key format")
            return None
    except ImportError:
        print("❌ python-dotenv not available. Please install it manually:")
        print("  pip install python-dotenv")
        return None

def create_database_safe(api_key):
    """Create the vector database with error handling."""
    print("🔧 Creating vector database...")
    
    try:
        # Set API key in environment
        os.environ['OPENAI_API_KEY'] = api_key
        
        # Try to import required modules
        try:
            from retriever import create_vectordb
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Some dependencies may be missing. Trying to continue...")
            return False
        
        print("📊 This may take a few minutes...")
        print("📡 Downloading EU AI Act document...")
        print("🔧 Processing and creating embeddings...")
        
        retriever, vector_store = create_vectordb()
        
        if retriever and vector_store:
            print("✅ Vector database created successfully")
            return True
        else:
            print("❌ Failed to create database")
            return False
            
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        print("This might be due to:")
        print("  - Network connectivity issues")
        print("  - OpenAI API rate limits")
        print("  - Missing dependencies")
        return False

def test_basic_imports():
    """Test if basic imports work."""
    print("🧪 Testing basic imports...")
    
    imports_to_test = [
        ("streamlit", "Streamlit web framework"),
        ("openai", "OpenAI API client"),
        ("langchain", "LangChain framework"),
        ("requests", "HTTP requests"),
        ("rank_bm25", "BM25 ranking algorithm")
    ]
    
    failed_imports = []
    
    for module, description in imports_to_test:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description} - MISSING")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️ Missing imports: {failed_imports}")
        print("Please install missing packages manually:")
        for module in failed_imports:
            print(f"  pip install {module}")
        return False
    
    print("✅ All basic imports successful")
    return True

def main():
    """Main setup function with dependency handling."""
    print_header("EU AI Act RAG Setup")
    
    print("🎯 This script will set up the RAG system with:")
    print("   - Hybrid search (semantic + keyword)")
    print("   - Query expansion for legal terminology")
    print("   - Smart reranking based on content type")
    print("   - Better chunking strategies")
    print("   - Advanced embeddings")
    
    # Step 1: Check Python version
    print_step(1, "Checking Python Version")
    if not check_python_version():
        return False
    
    # Step 2: Install dependencies
    print_step(2, "Installing Dependencies")
    if not install_dependencies():
        print("⚠️ Some dependencies failed to install, but continuing...")
    
    # Step 3: Test basic imports
    print_step(3, "Testing Basic Imports")
    if not test_basic_imports():
        print("❌ Critical imports failed. Please install missing packages manually.")
        return False
    
    # Step 4: Check API key
    print_step(4, "Checking OpenAI API Key")
    api_key = check_api_key()
    if not api_key:
        return False
    
    # Step 5: Create database
    print_step(5, "Creating Vector Database")
    db_created = create_database_safe(api_key)
    
    if not db_created:
        print("⚠️ Database creation failed.")
        print("You may need to run the database creation manually: python retriever.py")
        
        # Check if database exists
        if os.path.exists("faiss_vectordb"):
            print("✅ FAISS database found - you can use the system")
        else:
            print("❌ No vector database found. Please run: python retriever.py")
        
        return False
    
    # Success message
    print_header("Setup Complete! 🎉")
    print("✅ EU AI Act RAG system is ready!")
    print("\n🚀 Next Steps:")
    print("1. Run the app: streamlit run app.py")
    print("2. Try asking complex questions about the EU AI Act")
    print("\n💡 The system provides:")
    print("   - Superior retrieval quality with hybrid search")
    print("   - Automatic query expansion for legal terms")
    print("   - Smart reranking based on content relevance")
    print("   - Comprehensive answers")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup encountered issues.")
        print("\n🔧 Manual setup options:")
        print("1. Install missing packages: pip install rank-bm25")
        print("2. Create database: python retriever.py")
        print("3. Run system: streamlit run app.py")
        input("\nPress Enter to exit...")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")
        input("\nPress Enter to exit...")
        sys.exit(0) 