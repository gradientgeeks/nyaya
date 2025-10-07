#!/bin/bash
# Remote Training Deployment Script
# This script helps you set up and run training on a remote GPU

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - UPDATE THESE
REMOTE_USER="your_username"
REMOTE_HOST="your_remote_gpu_host"
REMOTE_PORT="22"
REMOTE_PATH="/home/${REMOTE_USER}/nyaya_training"
LOCAL_DATASET_PATH="/home/nyaya/server/dataset"
LOCAL_MODELS_PATH="/home/nyaya/server/src/models"

# Training configuration
BATCH_SIZE=32
NUM_EPOCHS=15
LEARNING_RATE=2e-5
CONTEXT_MODE="prev"
MODEL_TYPE="inlegalbert"

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Function to check if SSH connection works
check_connection() {
    print_header "Checking Remote Connection"
    if ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} "echo 'Connection successful'" > /dev/null 2>&1; then
        print_success "Successfully connected to ${REMOTE_HOST}"
        return 0
    else
        print_error "Failed to connect to ${REMOTE_HOST}"
        print_info "Please check your SSH credentials and try again"
        return 1
    fi
}

# Function to prepare local files
prepare_local_files() {
    print_header "Preparing Local Files"
    
    # Create temporary training package
    TEMP_DIR=$(mktemp -d)
    print_info "Creating training package in ${TEMP_DIR}"
    
    mkdir -p ${TEMP_DIR}/models
    mkdir -p ${TEMP_DIR}/dataset
    
    # Copy model files
    print_info "Copying model files..."
    cp -r ${LOCAL_MODELS_PATH}/* ${TEMP_DIR}/models/
    
    # Copy dataset
    print_info "Copying dataset..."
    cp -r ${LOCAL_DATASET_PATH}/* ${TEMP_DIR}/dataset/
    
    # Create training script
    cat > ${TEMP_DIR}/train.sh << 'EOF'
#!/bin/bash
cd $(dirname $0)/models/training

python train.py \
    --train_data ../../dataset/Hier_BiLSTM_CRF/train \
    --val_data ../../dataset/Hier_BiLSTM_CRF/val \
    --test_data ../../dataset/Hier_BiLSTM_CRF/test \
    --model_type MODEL_TYPE_PLACEHOLDER \
    --context_mode CONTEXT_MODE_PLACEHOLDER \
    --batch_size BATCH_SIZE_PLACEHOLDER \
    --num_epochs NUM_EPOCHS_PLACEHOLDER \
    --learning_rate LEARNING_RATE_PLACEHOLDER \
    --output_dir ./trained_models \
    --device cuda

echo "Training completed at $(date)"
EOF
    
    # Replace placeholders
    sed -i "s/MODEL_TYPE_PLACEHOLDER/${MODEL_TYPE}/g" ${TEMP_DIR}/train.sh
    sed -i "s/CONTEXT_MODE_PLACEHOLDER/${CONTEXT_MODE}/g" ${TEMP_DIR}/train.sh
    sed -i "s/BATCH_SIZE_PLACEHOLDER/${BATCH_SIZE}/g" ${TEMP_DIR}/train.sh
    sed -i "s/NUM_EPOCHS_PLACEHOLDER/${NUM_EPOCHS}/g" ${TEMP_DIR}/train.sh
    sed -i "s/LEARNING_RATE_PLACEHOLDER/${LEARNING_RATE}/g" ${TEMP_DIR}/train.sh
    
    chmod +x ${TEMP_DIR}/train.sh
    
    # Create requirements file
    cat > ${TEMP_DIR}/requirements.txt << EOF
torch>=2.0.0
transformers==4.35.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
tensorboard>=2.13.0
spacy>=3.6.0
EOF
    
    print_success "Training package prepared"
    echo ${TEMP_DIR}
}

# Function to transfer files to remote
transfer_files() {
    PACKAGE_DIR=$1
    print_header "Transferring Files to Remote GPU"
    
    print_info "Creating remote directory..."
    ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_PATH}"
    
    print_info "Transferring files (this may take a while)..."
    rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
        ${PACKAGE_DIR}/ \
        ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
    
    print_success "Files transferred successfully"
}

# Function to setup remote environment
setup_remote_env() {
    print_header "Setting Up Remote Environment"
    
    print_info "Creating Python virtual environment..."
    ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
cd REMOTE_PATH_PLACEHOLDER
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
EOF
    
    print_success "Remote environment setup complete"
}

# Function to start training
start_training() {
    print_header "Starting Training on Remote GPU"
    
    print_info "Starting training in screen session..."
    ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} << EOF
cd ${REMOTE_PATH}
screen -dmS nyaya_training bash -c "source venv/bin/activate && bash train.sh > training_\$(date +%Y%m%d_%H%M%S).log 2>&1"
EOF
    
    print_success "Training started in screen session 'nyaya_training'"
    print_info "Monitor training with: ssh ${REMOTE_USER}@${REMOTE_HOST} 'screen -r nyaya_training'"
    print_info "Detach from screen: Ctrl+A, then D"
}

# Function to check training status
check_status() {
    print_header "Checking Training Status"
    
    ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} << EOF
cd ${REMOTE_PATH}
if screen -list | grep -q nyaya_training; then
    echo "✅ Training session is running"
    echo ""
    echo "📊 Latest log entries:"
    tail -n 20 training_*.log 2>/dev/null || echo "No logs found yet"
    echo ""
    echo "🖥️  GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
else
    echo "⚠️  No training session found"
fi
EOF
}

# Function to download trained model
download_model() {
    print_header "Downloading Trained Model"
    
    LOCAL_OUTPUT_DIR="/home/nyaya/server/trained_models"
    mkdir -p ${LOCAL_OUTPUT_DIR}
    
    print_info "Downloading trained model files..."
    rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
        ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/models/training/trained_models/ \
        ${LOCAL_OUTPUT_DIR}/
    
    print_success "Model downloaded to ${LOCAL_OUTPUT_DIR}"
    
    # List downloaded files
    echo ""
    echo "📁 Downloaded files:"
    ls -lh ${LOCAL_OUTPUT_DIR}
}

# Function to cleanup remote
cleanup_remote() {
    print_header "Cleaning Up Remote Files"
    
    read -p "⚠️  This will delete training files on remote. Continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} "rm -rf ${REMOTE_PATH}"
        print_success "Remote files cleaned up"
    else
        print_info "Cleanup cancelled"
    fi
}

# Main menu
show_menu() {
    echo ""
    print_header "Remote GPU Training Helper"
    echo ""
    echo "1) Check connection"
    echo "2) Deploy and start training"
    echo "3) Check training status"
    echo "4) Download trained model"
    echo "5) Cleanup remote files"
    echo "6) Configure settings"
    echo "7) Exit"
    echo ""
}

configure_settings() {
    print_header "Configuration"
    
    echo "Current settings:"
    echo "  Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
    echo "  Path: ${REMOTE_PATH}"
    echo "  Batch size: ${BATCH_SIZE}"
    echo "  Epochs: ${NUM_EPOCHS}"
    echo "  Learning rate: ${LEARNING_RATE}"
    echo "  Context mode: ${CONTEXT_MODE}"
    echo ""
    
    read -p "Edit remote host? (current: ${REMOTE_HOST}) [Enter to skip]: " input
    [ -n "$input" ] && REMOTE_HOST=$input
    
    read -p "Edit remote user? (current: ${REMOTE_USER}) [Enter to skip]: " input
    [ -n "$input" ] && REMOTE_USER=$input
    
    read -p "Edit batch size? (current: ${BATCH_SIZE}) [Enter to skip]: " input
    [ -n "$input" ] && BATCH_SIZE=$input
    
    read -p "Edit number of epochs? (current: ${NUM_EPOCHS}) [Enter to skip]: " input
    [ -n "$input" ] && NUM_EPOCHS=$input
    
    print_success "Settings updated"
}

deploy_and_train() {
    check_connection || return 1
    
    PACKAGE_DIR=$(prepare_local_files)
    transfer_files ${PACKAGE_DIR}
    
    # Update setup command with actual path
    ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} << EOF
cd ${REMOTE_PATH}
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
EOF
    
    start_training
    
    # Cleanup temp directory
    rm -rf ${PACKAGE_DIR}
    
    print_success "Deployment complete!"
    echo ""
    print_info "Next steps:"
    echo "  - Monitor training: Use option 3 or SSH and run 'screen -r nyaya_training'"
    echo "  - Download model: Use option 4 after training completes"
}

# Main script
main() {
    while true; do
        show_menu
        read -p "Select option: " choice
        
        case $choice in
            1) check_connection ;;
            2) deploy_and_train ;;
            3) check_status ;;
            4) download_model ;;
            5) cleanup_remote ;;
            6) configure_settings ;;
            7) print_info "Goodbye!"; exit 0 ;;
            *) print_error "Invalid option" ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi
