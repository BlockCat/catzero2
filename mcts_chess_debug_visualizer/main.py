from flask import Flask, render_template, jsonify, send_from_directory, request
import os
import json
import re
from pathlib import Path


app = Flask(__name__)

PATH = "/mnt/D012EBD012EBB99C/Workspace/projects/alphazero/exporter"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/tree')
def get_tree():
    """Get the directory tree structure with states"""
    tree = {}
    
    if not os.path.exists(PATH):
        return jsonify(tree)
    
    # Iterate through m1, m2, etc.
    for model_dir in sorted(os.listdir(PATH)):
        model_path = os.path.join(PATH, model_dir)
        if not os.path.isdir(model_path):
            continue
            
        tree[model_dir] = {}
        
        # Iterate through timestamp folders
        for timestamp_dir in sorted(os.listdir(model_path)):
            timestamp_path = os.path.join(model_path, timestamp_dir)
            if not os.path.isdir(timestamp_path):
                continue
                
            tree[model_dir][timestamp_dir] = {}
            
            # Check if there are subdirectories (like the hash folders)
            subdirs = [d for d in os.listdir(timestamp_path) 
                      if os.path.isdir(os.path.join(timestamp_path, d))]
            
            if subdirs:
                # Has subdirectories
                for subdir in sorted(subdirs):
                    subdir_path = os.path.join(timestamp_path, subdir)
                    states = get_states_from_directory(subdir_path)
                    if states:
                        tree[model_dir][timestamp_dir][subdir] = states
            else:
                # No subdirectories, debug files are directly in timestamp folder
                states = get_states_from_directory(timestamp_path)
                if states:
                    tree[model_dir][timestamp_dir] = states
    
    return jsonify(tree)

def get_states_from_directory(directory):
    """Get all debug files and their states from a directory"""
    state = {}
    
    # Find all debug_*.json files
    counter = 0
    for filename in os.listdir(directory):
        match = re.match(r'debug_(\d+)\.json', filename)
        if match:
            counter += 1
    
    if counter == 0:
        return None
    


    # Sort by iteration number
    filepath = os.path.join(directory, 'debug_0.json')
    with open(filepath, 'r') as f:
        data = json.load(f)        
        state['state'] = data.get('state', 'Unknown')
    state['iterations'] = counter
    
    return state

@app.route('/api/data/<path:filepath>')
def get_data(filepath):

    """Get the data for a specific debug file"""
    full_path = os.path.join(PATH, filepath)
    
    iteration = request.args.get('iteration')
    if iteration is not None:
        full_path = os.path.join(full_path, f'debug_{iteration}.json')
    else:
        full_path = os.path.join(full_path, 'debug_0.json')
    
    if not os.path.exists(full_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        with open(full_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=5000)