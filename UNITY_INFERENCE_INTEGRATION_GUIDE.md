# Unity Inference Integration Guide

## 🔍 **What the Inference Policy Takes as Input**

The policy needs a **complete situational awareness picture** - basically everything a human pilot would need to make intercept decisions:

### **Core Input Categories**:

1. **📍 My State (Blue/Interceptor)**:
   - Position: `[x, y, z]` in meters
   - Velocity: `[vx, vy, vz]` in m/s  
   - Orientation: `[w, x, y, z]` quaternion
   - Angular velocity: `[pitch_rate, yaw_rate, roll_rate]` rad/s
   - **Fuel remaining**: 0.0-1.0 fraction

2. **🎯 Target State (Red/Threat)**:
   - Position: `[x, y, z]` in meters
   - Velocity: `[vx, vy, vz]` in m/s
   - Orientation: `[w, x, y, z]` quaternion

3. **📡 Guidance Information** (derived from above):
   - **Line-of-sight vector**: Unit vector pointing at target
   - **Range**: Distance to target in meters
   - **Closing speed**: How fast we're approaching (negative = diverging)
   - **Line-of-sight rate**: How fast the target direction is changing
   - **Field-of-view status**: Can sensors still track target?

4. **🌍 Environment**:
   - Wind velocity: `[wx, wy, wz]` m/s
   - Sensor noise levels
   - Mission time elapsed

## 🎮 **Unity Integration Architecture**

Here's how it would work with Unity doing physics:

```csharp
// Unity Update Loop (every frame or at guidance frequency)
void Update() {
    // 1. Gather current states from Unity physics
    var interceptorState = GetInterceptorState();  // Your Unity missile
    var threatState = GetThreatState();            // Target trajectory
    
    // 2. Send to inference API
    var request = new InferenceRequest {
        blue = interceptorState,    // Your missile state
        red = threatState,          // Target state  
        guidance = CalculateGuidance(interceptorState, threatState),
        env = GetEnvironmentState()
    };
    
    var response = await CallInferenceAPI(request);
    
    // 3. Apply control commands to Unity physics
    ApplyControlCommands(response.action);
}

void ApplyControlCommands(ActionCommand cmd) {
    // Apply thrust to missile Rigidbody
    rigidbody.AddForce(transform.forward * cmd.thrust_cmd * maxThrust);
    
    // Apply angular rate commands (convert to torques)
    var targetAngularVel = new Vector3(
        cmd.rate_cmd_radps.pitch,
        cmd.rate_cmd_radps.yaw, 
        cmd.rate_cmd_radps.roll
    );
    
    // Apply torques to achieve desired angular rates
    ApplyAngularRateControl(targetAngularVel);
    
    // Update fuel
    currentFuel -= cmd.thrust_cmd * fuelConsumptionRate * Time.deltaTime;
}
```

## 📡 **The Key Insight: Attacker/Threat**

For the **attacker/threat (red)**, you have several options:

### **Option A: Pre-programmed Trajectory**
```csharp
// Simple ballistic trajectory - no AI needed
Vector3 GetThreatState(float time) {
    return initialPosition + initialVelocity * time + 0.5f * gravity * time * time;
}
```

### **Option B: Scripted Maneuvers**  
```csharp
// More complex evasive maneuvers
Vector3 GetThreatState(float time) {
    // Weaving pattern, altitude changes, etc.
    return CalculateEvasiveTrajectory(time);
}
```

### **Option C: Dual AI (Advanced)**
```csharp
// Run separate AI for the threat (different model)
var threatResponse = await CallThreatInferenceAPI(threatRequest);
ApplyThreatCommands(threatResponse.action);
```

## 🔄 **Complete Flow Example**

1. **Unity Frame Start**:
   - Interceptor at `[100, 50, 1000]`, velocity `[150, 0, 0]`, fuel `0.85`
   - Threat at `[5000, 100, 1200]`, velocity `[-80, -20, -5]`

2. **Calculate Guidance**:
   - Range: `4901m`
   - Line-of-sight: `[0.98, 0.01, 0.04]` 
   - Closing speed: `230 m/s`

3. **Send to Inference**:
   ```json
   {
     "blue": {"pos_m": [100, 50, 1000], "vel_mps": [150, 0, 0], "fuel_frac": 0.85},
     "red": {"pos_m": [5000, 100, 1200], "vel_mps": [-80, -20, -5]},
     "guidance": {"range_m": 4901, "closing_speed_mps": 230}
   }
   ```

4. **Get Control Response**:
   ```json
   {
     "action": {
       "thrust_cmd": 0.75,
       "rate_cmd_radps": {"pitch": 0.12, "yaw": -0.05, "roll": 0.02}
     }
   }
   ```

5. **Unity Physics Integration**:
   - Apply 75% thrust forward
   - Apply angular rates for steering
   - Update fuel consumption
   - Let Unity physics handle acceleration, rotation, etc.

## 🎯 **Benefits of Unity Physics**

1. **Real-time smooth motion** (60fps interpolation)
2. **Unity handles collisions, constraints, etc.**
3. **Visual debugging** (see forces, trajectories)
4. **Flexible physics tuning** (drag, mass, etc.)
5. **Natural integration** with Unity's systems

## 📝 **What You Need**

1. **Inference API call** (HTTP request/response)
2. **State extraction** (position, velocity from Unity objects)
3. **Control application** (thrust + angular rate → forces/torques)
4. **Threat trajectory** (ballistic, scripted, or AI)

The policy doesn't care about Unity vs. server physics - it just needs current states and provides control commands!

## 🚀 **Recommended Implementation Steps**

### **Phase 1: Basic Integration**
1. Create Unity missile GameObject with Rigidbody
2. Implement HTTP client for inference API calls
3. Extract state from Unity physics (position, velocity, etc.)
4. Apply thrust commands as forces
5. Test with simple ballistic threat

### **Phase 2: Advanced Control**
1. Implement proper angular rate control (PID controllers)
2. Add fuel system with realistic consumption
3. Implement coordinate system transformations (ENU ↔ Unity)
4. Add guidance calculations (LOS, closing speed, etc.)

### **Phase 3: Production Features**
1. Add multiple threat types and scenarios
2. Implement error handling and fallback behaviors
3. Add telemetry and performance monitoring  
4. Create replay/analysis tools

## 🔧 **Technical Considerations**

### **Update Frequency**
- **Inference calls**: 1-2 Hz (realistic guidance frequency)
- **Physics integration**: 50-60 Hz (Unity FixedUpdate)
- **Visual updates**: 60+ Hz (Unity Update)

### **Coordinate Systems**
- **Server expects**: ENU (East-North-Up, right-handed)
- **Unity uses**: Left-handed coordinate system
- **Solution**: Transform coordinates in API calls

### **Error Handling**
- Network timeouts → Use last known command
- Invalid responses → Safe fallback behavior
- Physics instabilities → Command limiting/smoothing

### **Performance**
- Async API calls (don't block Unity main thread)
- Command interpolation between inference updates
- Efficient state serialization/deserialization

## 💡 **Recommended Architecture: Hybrid Server-Unity Approach**

### **🔄 Optimal Flow Design:**

```
1. Unity → Current interceptor state → Server
2. Server → Combines with predefined threat trajectory  
3. Server → Sends complete situation to AI model
4. Server → Gets control commands from model
5. Server → Returns BOTH threat state + control commands → Unity
6. Unity → Displays threat, applies commands to interceptor physics
```

## ✅ **Why This Architecture Is Excellent:**

### **1. Server Handles "Mission Planning"**
- **Predefined scenarios** (easy/medium/hard threats)
- **Deterministic threat trajectories** 
- **Consistent guidance calculations**
- **No Unity coding for complex guidance math**

### **2. Unity Handles "Execution"** 
- **Real-time 6DOF physics** for the interceptor
- **Smooth visual representation** 
- **Natural integration** with Unity's systems
- **Flexible physics tuning** (drag, mass, etc.)

### **3. Clean Separation of Concerns**
- **Server**: Mission logic, AI inference, threat behavior
- **Unity**: Visualization, interceptor physics, user interaction

## 🎯 **Implementation Details:**

### **Server Side (Leverages Existing Code):**
```python
# Server receives Unity interceptor state
request = {
    "blue": unity_interceptor_state,  # From Unity physics
    "red": get_predefined_threat_state(scenario, time),  # Server generates
    "guidance": calculate_guidance(blue, red),  # Server computes
}

# AI generates commands
commands = model.predict(observation)

# Return both to Unity
response = {
    "action": commands,  # For interceptor physics
    "threat_state": red,  # For threat visualization
    "diagnostics": {...}  # For debugging/UI
}
```

### **Unity Side:**
```csharp
void FixedUpdate() {
    // 1. Get current interceptor state from Unity physics
    var interceptorState = ExtractPhysicsState(interceptorRigidbody);
    
    // 2. Send to server (at 1-2 Hz, not every frame)
    if (Time.time > nextInferenceTime) {
        var response = await CallServer(interceptorState);
        
        // 3. Update threat visualization
        UpdateThreatDisplay(response.threat_state);
        
        // 4. Store commands for physics application
        currentCommands = response.action;
        nextInferenceTime = Time.time + 0.5f; // 2 Hz
    }
    
    // 5. Apply commands to interceptor physics (every physics frame)
    ApplyControlCommands(currentCommands);
}
```

## 🚀 **This Architecture Is Perfect Because:**

1. **✅ Leverages existing server logic** (already has scenarios, guidance)
2. **✅ Unity gets both states** (interceptor commands + threat position) 
3. **✅ Real-time performance** (Unity physics at 50Hz, inference at 2Hz)
4. **✅ Easy debugging** (server logs, Unity visual feedback)
5. **✅ Scalable** (add new scenarios server-side)

## 💪 **Implementation Confidence: High**

This is a **production-ready architecture** used in real missile guidance systems. You're essentially creating:
- **Server**: "Mission computer" (guidance, threats, AI)
- **Unity**: "Vehicle dynamics + display" (physics, visualization)

The division of labor is perfect - each system does what it's best at!

## 🔧 **Key Advantages of This Approach:**

### **For Development:**
- **Rapid iteration** (change scenarios server-side)
- **Easy testing** (server can run without Unity)
- **Clear debugging** (separate logs for mission vs. physics)

### **For Performance:**
- **Efficient network usage** (2Hz vs 60Hz updates)
- **Smooth visuals** (Unity interpolation between commands)
- **Realistic physics** (proper 6DOF integration)

### **For Scalability:**
- **Multiple Unity clients** can connect to same server
- **Different threat scenarios** without Unity changes
- **A/B testing** of different AI models server-side