# Flask API Research for Unity-RL Bridge

## Core API Patterns

### JSON Request/Response Handling

Flask provides built-in JSON support with several key patterns:

#### JSON Request Parsing
```python
# Automatic JSON parsing
@app.post("/user/<int:id>")
def user_update(id):
    user = User.query.get_or_404(id)
    user.update_from_json(request.json)  # request.json automatically parses JSON
    db.session.commit()
    return user.to_json()
```

#### JSON Response Generation
```python
# Automatic JSON response - return dict/list directly
@app.route("/me")
def me_api():
    user = get_current_user()
    return {
        "username": user.username,
        "theme": user.theme,
        "image": url_for("user_image", filename=user.image)
    }

# Or use jsonify() for complex responses
@app.route("/users")
def user_list():
    users = User.query.order_by(User.name).all()
    return jsonify([u.to_json() for u in users])
```

### Route Configuration

#### HTTP Method Specification
```python
# Multiple methods
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return do_the_login()
    else:
        return show_the_login_form()

# Shortcut decorators (Flask 2.0+)
@app.get('/data')
def get_data():
    return {"data": "value"}

@app.post('/data')
def post_data():
    return {"received": request.json}
```

### Error Handling

#### Custom Error Handlers
```python
@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404

# For APIs, return JSON errors
@app.errorhandler(404)
def api_not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500
```

### Request Lifecycle Hooks

```python
@app.before_request
def before_request():
    # Run code before each request
    pass

@app.after_request
def after_request(response):
    # Modify response before sending
    return response

@app.teardown_request
def teardown_request(exception):
    # Clean up resources after request
    pass
```

## CORS Configuration (Flask-CORS)

### Basic Setup
```python
from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Or with specific configuration
cors = CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://unity-client"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
```

### Route-Specific CORS
```python
@app.route("/api/data")
@cross_origin(origins=["http://localhost:3000"])
def get_data():
    return {"data": "value"}
```

### Unity Integration Considerations
- Unity will need CORS enabled for web requests
- Configure allowed origins to include Unity client addresses
- Enable preflight request handling for complex requests

### Security Notes
- CORS disables cookie submission across domains by default
- Add CSRF protection when enabling credentials
- Be specific with origins in production (avoid "*")

## Performance and Production

### Configuration Management
```python
app.config['JSON_SORT_KEYS'] = False  # Don't sort JSON keys
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Compact JSON in production
```

### Request Validation
```python
@app.route('/api/endpoint', methods=['POST'])
def endpoint():
    # Check content type
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    # Get JSON data
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400
    
    # Validate required fields
    if 'required_field' not in data:
        return jsonify({'error': 'Missing required_field'}), 400
    
    return jsonify({'success': True})
```

### Testing Support
```python
# Test client for unit tests
client = app.test_client()
response = client.post('/api/endpoint', 
                      json={'data': 'value'},
                      content_type='application/json')
```

## Key Takeaways for Unity-RL Bridge

1. **Use Flask's automatic JSON handling** - Return dicts directly for simple responses
2. **Implement comprehensive error handlers** - Return structured JSON errors with proper HTTP codes
3. **Configure CORS properly** - Enable CORS for Unity client communication
4. **Validate requests thoroughly** - Check content type, JSON validity, and required fields
5. **Use route-specific decorators** - `@app.post()` is cleaner than `methods=['POST']`
6. **Structure responses consistently** - Use standard format for all API responses