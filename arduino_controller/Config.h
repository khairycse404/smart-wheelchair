#ifndef WHEELCHAIR_CONFIG_H
#define WHEELCHAIR_CONFIG_H

#include <Arduino.h>

// ============================================================================
// 1. SERIAL & TIMING CONFIGURATION
// ============================================================================
#define SERIAL_BAUD_RATE        115200
#define GPS_DEFAULT_BAUD_RATE   9600
#define COMMAND_WATCHDOG_MS     1200   // Auto-neutral if Mini PC silent > 1.2s
#define TELEMETRY_INTERVAL_MS   100    // 10 Hz status report back to Mini PC
#define RAMP_INTERVAL_MS        15     // Smooth DAC ramp update step (ms)
#define RAMP_MAX_STEP_DAC       45     // Max DAC counts change per 15ms step

// ============================================================================
// 2. DEFAULT ARDUINO PIN ASSIGNMENTS (Runtime Customizable from Web GUI)
// ============================================================================
// Note: I2C MCP4725 DACs also use hardware SDA (A4) and SCL (A5) on Arduino Uno/Nano
#define PIN_SPEED_DEFAULT       9      // Y-Axis (Forward/Reverse) PWM / Aux Output
#define PIN_STEER_DEFAULT       10     // X-Axis (Left/Right) PWM / Aux Output
#define PIN_RELAY_DEFAULT       7      // Joystick Takeover Relay (HIGH = Arduino DAC active)
#define PIN_ESTOP_DEFAULT       2      // Hardware E-Stop Switch Input (INPUT_PULLUP)
#define PIN_GPS_RX_DEFAULT      4      // Arduino RX <- Connects to GPS TX
#define PIN_GPS_TX_DEFAULT      3      // Arduino TX -> Connects to GPS RX

// ============================================================================
// 3. DUAL MCP4725 12-BIT I2C DAC CONFIGURATION (Wheelchair Joystick Spoofing)
// ============================================================================
// Wheelchair joysticks operate around a 2.50V center neutral reference:
//   - 2.50V (DAC 2048) = Neutral / Motors Stopped (Brakes Locked)
//   - 0.50V..2.50V     = Reverse / Left
//   - 2.50V..4.50V     = Forward / Right
#define DAC_I2C_ADDR_SPEED      0x60   // MCP4725 #1: Y-Axis (Forward / Backward)
#define DAC_I2C_ADDR_STEER      0x61   // MCP4725 #2: X-Axis (Left / Right)

#define DAC_RESOLUTION_MAX      4095
#define DAC_NEUTRAL             2048   // 2.50V exact center neutral
#define DAC_MIN_SAFE            410    // ~0.50V lower joystick limit (avoids fault trip)
#define DAC_MAX_SAFE            3686   // ~4.50V upper joystick limit (avoids fault trip)
#define DAC_MAX_SWING           1638   // Max deviation from 2048 neutral (2048 +/- 1638)

// ============================================================================
// 4. OPERATING MODES
// ============================================================================
enum DriveMode : uint8_t {
    MODE_STOP       = 0,
    MODE_MANUAL     = 1,
    MODE_TEST       = 2,
    MODE_AUTONOMOUS = 3
};

// ============================================================================
// 5. MOTOR TEST ACTIONS
// ============================================================================
enum TestAction : uint8_t {
    TEST_NONE       = 0,
    TEST_FWD        = 1,
    TEST_BACK       = 2,
    TEST_LEFT       = 3,
    TEST_RIGHT      = 4,
    TEST_NEUTRAL    = 5,
    TEST_SWEEP      = 6
};

#endif // WHEELCHAIR_CONFIG_H
