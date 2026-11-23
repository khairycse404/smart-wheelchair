#ifndef DAC_MOTOR_CONTROLLER_H
#define DAC_MOTOR_CONTROLLER_H

#include <Arduino.h>
#include <Wire.h>
#include "Config.h"

class DacMotorController {
public:
    DacMotorController();

    // Initializes I2C bus, DACs to 2.50V neutral, and digital/PWM pins
    void begin();

    // Live reconfiguration of pins and motor reversal polarity from the Web GUI
    void configureHardware(
        uint8_t speedPin,
        uint8_t steerPin,
        uint8_t relayPin,
        uint8_t estopPin,
        bool revDriveMotor,
        bool revSteerMotor
    );

    // Sets desired mode, speed percentage (-100 to +100), steer angle (-75 to +75), and software E-Stop
    void setCommand(DriveMode mode, int16_t speedPct, int16_t steerDeg, bool swEstop);

    // Starts a timed pre-autonomous motor test sequence
    void triggerTest(TestAction action);

    // Non-blocking periodic update: checks hardware E-Stop, test timers, and ramps DAC outputs
    void update();

    // Immediately forces neutral 2.50V and disengages takeover relay
    void emergencyNeutral();

    // Getters for live GUI telemetry
    DriveMode getMode() const { return _mode; }
    int16_t getCommandSpeedPct() const { return _targetSpeedPct; }
    int16_t getCommandSteerDeg() const { return _targetSteerDeg; }
    uint16_t getCurrentDacSpeed() const { return _currentDacSpeed; }
    uint16_t getCurrentDacSteer() const { return _currentDacSteer; }
    float getVoltageSpeed() const { return (_currentDacSpeed * 5.0f) / 4095.0f; }
    float getVoltageSteer() const { return (_currentDacSteer * 5.0f) / 4095.0f; }
    bool isRelayEngaged() const { return _relayEngaged; }
    bool isEstopActive() const { return _swEstop || _hwEstop; }
    uint8_t getPinSpeed() const { return _pinSpeed; }
    uint8_t getPinSteer() const { return _pinSteer; }
    uint8_t getPinRelay() const { return _pinRelay; }
    uint8_t getPinEstop() const { return _pinEstop; }
    bool isDriveReversed() const { return _reverseDrive; }
    bool isSteerReversed() const { return _reverseSteer; }
    TestAction getActiveTest() const { return _activeTest; }

private:
    uint8_t _pinSpeed;
    uint8_t _pinSteer;
    uint8_t _pinRelay;
    uint8_t _pinEstop;

    bool _reverseDrive;
    bool _reverseSteer;

    DriveMode _mode;
    int16_t _targetSpeedPct;
    int16_t _targetSteerDeg;
    bool _swEstop;
    bool _hwEstop;
    bool _relayEngaged;

    uint16_t _targetDacSpeed;
    uint16_t _targetDacSteer;
    uint16_t _currentDacSpeed;
    uint16_t _currentDacSteer;
    unsigned long _lastRampMs;

    TestAction _activeTest;
    unsigned long _testStartMs;

    void _recomputeDacTargets(int16_t speedPct, int16_t steerDeg);
    void _writeDacChannel(uint8_t i2cAddr, uint16_t value12bit);
    void _applyOutputs();
    void _updateTestSequence();
};

#endif // DAC_MOTOR_CONTROLLER_H
