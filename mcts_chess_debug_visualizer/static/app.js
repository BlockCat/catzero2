let currentData = null;
let currentPath = null;
let allStates = 0;
let currentIndex = -1;
let board = null;
let game = null;

// Initialize the application
$(document).ready(function() {
    loadTree();
    initializeBoard();
    setupEventListeners();
});

function initializeBoard() {
    board = Chessboard('board', {
        draggable: false,
        position: 'start',
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
    });
    game = new Chess();
}

function setupEventListeners() {
    $('#prev-btn').click(loadPrevious);
    $('#next-btn').click(loadNext);
    $('#goto-btn').click(goToIteration);
    $('#goto-input').on('keypress', function(e) {
        if (e.which === 13) { // Enter key
            goToIteration();
        }
    });
}

// Load the directory tree
function loadTree() {
    console.log('Loading tree...');
    $.get('/api/tree', function(tree) {
        console.log('Tree loaded:', tree);
        renderTreeNavigation(tree);
    }).fail(function(xhr, status, error) {
        console.error('Failed to load tree:', error);
        $('#tree-view').html('<p style="color: #f14c4c;">Error loading tree</p>');
    });
}

function renderTreeNavigation(tree) {
    const $container = $('#tree-view');
    $container.empty();
    
    const entries = Object.entries(tree);
    console.log('Rendering tree with entries:', entries);
    
    if (entries.length === 0) {
        $container.html('<p style="color: #858585; padding: 10px;">No data found</p>');
        return;
    }
    
    for (const [modelName, timestamps] of entries) {
        const $modelNode = createFolderNode(modelName);
        $container.append($modelNode);
        
        const $modelChildren = $('<div class="tree-children"></div>');
        $container.append($modelChildren);
        
        for (const [timestamp, content] of Object.entries(timestamps)) {
            const $timestampNode = createFolderNode(timestamp);
            $modelChildren.append($timestampNode);
            
            const $timestampChildren = $('<div class="tree-children"></div>');
            $modelChildren.append($timestampChildren);
            
            // Check if content is an array of states or an object with subdirectories
            console.log('Rendering timestamp:', timestamp, 'Content:', content);
            if (Array.isArray(content)) {
                // Direct states
                renderStates($timestampChildren, content, `${modelName}/${timestamp}`);
            } else {
                // Has subdirectories
                for (const [subdir, states] of Object.entries(content)) {                                       
                    renderStates($timestampChildren, states, `${modelName}/${timestamp}/${subdir}`);
                }
            }
        }
    }
}

function createFolderNode(name) {
    const $node = $('<div class="tree-node folder"></div>');
    $node.html(`<span class="toggle">▶</span>${name}`);
    
    $node.click(function(e) {
        e.stopPropagation();
        const $children = $(this).next('.tree-children');
        $children.toggleClass('expanded');
        
        const $toggle = $(this).find('.toggle');
        $toggle.text($children.hasClass('expanded') ? '▼' : '▶');
    });
    
    return $node;
}

function renderStates($container, state, basePath) {
    console.log('Rendering states for:', basePath, 'Count:', state);
    
        const $stateNode = $('<div class="tree-node state"></div>');
        
        // Show a shortened version of the FEN
        const fenShort = shortenFEN(state.state);
        $stateNode.text(`${fenShort}`);
        $stateNode.attr('data-base-path', basePath);
        
        $stateNode.click(function(e) {
            e.stopPropagation();                       
            
            loadState(basePath, 0);

            allStates = state.iterations;
            
            // Update active state
            $('.tree-node.state').removeClass('active');
            $(this).addClass('active');
        });
        
        $container.append($stateNode);
    
}

function shortenFEN(fen) {
    // FEN format: position active_color castling en_passant halfmove fullmove
    // We'll just show the position part, truncated
    const parts = fen.split(' ');
    const position = parts[0];
    if (position.length > 25) {
        return position.substring(0, 25) + '...';
    }
    return position;
}

function loadState(path, iteration) {
    $.get(`/api/data/${path}?iteration=${iteration}`, function(data) {
        currentData = data;
        currentPath = path;
        currentIndex = iteration;
        
        updateBoard(data.state);
        renderTree(data);
        updateNavigationButtons();
        updateIterationInfo();
    }).fail(function() {
        alert('Error loading state data');
    });
}

function updateBoard(fen) {
    game = new Chess(fen);
    board.position(fen);
    
    $('#board-info').html(`
        <div><strong>FEN:</strong> ${fen}</div>
        <div><strong>Turn:</strong> ${game.turn() === 'w' ? 'White' : 'Black'}</div>
        <div><strong>In Check:</strong> ${game.in_check() ? 'Yes' : 'No'}</div>
    `);
}

function renderTree(data) {
    const nodes = buildTreeStructure(data);
    visualizeTree(nodes, data);
}

function buildTreeStructure(data) {
    const nodes = [];
    const n = data.actions.length;
    
    for (let i = 0; i < n; i++) {
        const node = {
            id: i,
            action: data.actions[i],
            visits: data.visit_counts[i],
            reward: data.rewards[i],
            policy: data.policy[i],
            children: [],
            parent: null
        };
        
        // Build children
        const childStart = data.children_start_index[i];
        const childCount = data.children_count[i];
        
        if (childStart !== null && childCount > 0) {
            for (let j = 0; j < childCount; j++) {
                const childId = childStart + j;
                node.children.push(childId);
            }
        }
        
        nodes.push(node);
    }
    
    // Set parent references
    nodes.forEach((node, i) => {
        node.children.forEach(childId => {
            if (nodes[childId]) {
                nodes[childId].parent = i;
            }
        });
    });
    
    return nodes;
}

function visualizeTree(nodes, data) {
    const svg = d3.select('#tree-svg');
    svg.selectAll('*').remove();
    
    const width = 2000;
    const height = Math.max(1000, nodes.length * 20);
    
    svg.attr('width', width)
       .attr('height', height);
    
    const g = svg.append('g')
                 .attr('transform', 'translate(50, 50)');
    
    // Create tree layout
    const root = d3.hierarchy({children: [buildHierarchy(nodes, 0)]});
    const treeLayout = d3.tree().size([height - 100, width - 200]);
    
    treeLayout(root);
    
    // Draw links
    const link = g.selectAll('.link')
        .data(root.links())
        .enter()
        .append('path')
        .attr('class', d => {
            const sourceId = d.source.data.id;
            const targetId = d.target.data.id;
            return isInChosenPath(sourceId, targetId, data.chosen_path) ? 'link highlighted' : 'link';
        })
        .attr('d', d3.linkHorizontal()
            .x(d => d.y)
            .y(d => d.x));
    
    // Draw edge labels
    g.selectAll('.edge-label')
        .data(root.links())
        .enter()
        .append('text')
        .attr('class', 'edge-label')
        .attr('x', d => (d.source.y + d.target.y) / 2)
        .attr('y', d => (d.source.x + d.target.x) / 2 - 5)
        .text(d => {
            const node = d.target.data;
            const action = node.action || '';
            const policy = (node.policy * 100).toFixed(1);
            return `${action} (${policy}%)`;
        });
    
    // Draw nodes
    const node = g.selectAll('.node')
        .data(root.descendants())
        .enter()
        .append('g')
        .attr('class', d => {
            let classes = 'node';
            if (data.chosen_path.includes(d.data.id)) {
                classes += ' highlighted';
            }
            if (d.data.id === data.chosen_node) {
                classes += ' chosen';
            }
            return classes;
        })
        .attr('transform', d => `translate(${d.y},${d.x})`)
        .on('click', function(event, d) {
            handleNodeClick(d.data, data);
        });
    
    node.append('circle')
        .attr('r', 8);
    
    node.append('text')
        .attr('dy', -12)
        .attr('text-anchor', 'middle')
        .text(d => `V:${d.data.visits ?? 0}`);
    
    node.append('text')
        .attr('dy', 20)
        .attr('text-anchor', 'middle')
        .text(d => `R:${d.data.reward?.toFixed(2) ?? 0 / d.data.visits ?? 1}`);
}

function buildHierarchy(nodes, nodeId) {
    const node = nodes[nodeId];
    const result = {
        id: nodeId,
        action: node.action,
        visits: node.visits,
        reward: node.reward,
        policy: node.policy,
        children: []
    };
    
    node.children.forEach(childId => {
        if (nodes[childId]) {
            result.children.push(buildHierarchy(nodes, childId));
        }
    });
    
    return result;
}

function isInChosenPath(sourceId, targetId, chosenPath) {
    const sourceIdx = chosenPath.indexOf(sourceId);
    const targetIdx = chosenPath.indexOf(targetId);
    return sourceIdx !== -1 && targetIdx !== -1 && targetIdx === sourceIdx + 1;
}

function handleNodeClick(nodeData, data) {
    // Find the path from root to this node
    const path = findPathToNode(nodeData.id, data);
    
    // Play the actions to get to this position
    const rootFEN = data.state;
    game = new Chess(rootFEN);
    
    for (let i = 1; i < path.length; i++) {
        const action = data.actions[path[i]];
        if (action) {
            game.move(action, {sloppy: true});
        }
    }
    
    board.position(game.fen());
    $('#board-info').html(`
        <div><strong>Node:</strong> ${nodeData.id}</div>
        <div><strong>Visits:</strong> ${nodeData.visits}</div>
        <div><strong>Reward:</strong> ${nodeData.reward.toFixed(3)}</div>
        <div><strong>FEN:</strong> ${game.fen()}</div>
        <div><strong>Turn:</strong> ${game.turn() === 'w' ? 'White' : 'Black'}</div>
    `);
}

function findPathToNode(nodeId, data) {
    const path = [nodeId];
    let current = nodeId;
    
    while (current !== 0) {
        let found = false;
        for (let i = 0; i < data.actions.length; i++) {
            const childStart = data.children_start_index[i];
            const childCount = data.children_count[i];
            
            if (childStart !== null && childCount > 0) {
                for (let j = 0; j < childCount; j++) {
                    if (childStart + j === current) {
                        path.unshift(i);
                        current = i;
                        found = true;
                        break;
                    }
                }
            }
            if (found) break;
        }
        if (!found) break;
    }
    
    return path;
}

function loadPrevious() {
    if (currentIndex > 0) {
        const prevState = allStates[currentIndex - 1];
        loadState(prevState.path, currentIndex - 1);
        
        // Update active state in tree
        $(`.tree-node.state[data-index="${currentIndex - 1}"]`).click();
    }
}

function loadNext() {
    if (currentIndex < allStates - 1) {
        const nextState = currentIndex + 1;
        loadState(currentPath, currentIndex + 1);
        
        // Update active state in tree
        $(`.tree-node.state[data-index="${currentIndex + 1}"]`).click();
    }
}

function goToIteration() {
    const targetIteration = parseInt($('#goto-input').val());
    
    if (isNaN(targetIteration) || targetIteration < 0) {
        alert('Please enter a valid iteration number');
        return;
    }
    
    if (targetIteration >= allStates) {
        alert(`Iteration ${targetIteration} does not exist. Maximum iteration is ${allStates - 1}`);
        return;
    }
    
    loadState(currentPath, targetIteration);
}

function updateNavigationButtons() {
    const hasData = currentPath !== null && allStates > 0;
    
    $('#prev-btn').prop('disabled', currentIndex <= 0 || !hasData);
    $('#next-btn').prop('disabled', currentIndex >= allStates - 1 || !hasData);
    $('#goto-btn').prop('disabled', !hasData);
    $('#goto-input').prop('disabled', !hasData);
    
    if (hasData) {
        $('#goto-input').attr('max', allStates - 1);
    }
}

function updateIterationInfo() {
    if (currentData) {
        $('#iteration-info').text(`Iteration ${currentData.iteration} (${currentIndex + 1}/${allStates})`);
    } else {
        $('#iteration-info').text('No iteration loaded');
    }
}
