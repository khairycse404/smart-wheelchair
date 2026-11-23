#include "DacMotorController.h"

DacMotorController::DacMotorController()
    : _pinSpeed(PIN_SPEED_DEFAULT),
      _pinSteer(PIN_STEER_DEFAULT),
      _pinRelay(PIN_RELAY_DEFAULT),
      _pinEstop(PIN_ESTOP_DEFAULT),
      _reverseDrive(false),
      _reverseSteer(false),
      _mode(MODE_STOP),
      _targetSpeedPct(0),
      _targetSteerDeg(0),
      _swEstop(false),
      _hwEstop(false),
      _relayEngaged(false),
      _targetDacSpeed(DAC_NEUTRAL),
      _targetDacSteer(DAC_NEUTRAL),
      _currentDacSpeed(DAC_NEUTRAL),
      _currentDacSteer(DAC_NEUTRAL),
      _lastRampMs(0),
      _activeTest(TEST_NONE),
      _testStartMs(0) {}

void DacMotorController::begin() {
    Wire.begin();
    Wire.setClock(400000L); // 400 kHz Fast I2C for low-latency DAC updates

    pinMode(_pinSpeed, OUTPUT);
    pinMode(_pinSteer, OUTPUT);
    pinMode(_pinRelay, OUTPUT);
    pinMode(_pinEstop, INPUT_PULLUP);

    // SAFETY FIRST: Initialize DACs to exact 2.50V center neutral BEFORE relay can ever engage
    emergencyNeutral();
}

void DacMotorController::configureHardware(
    uint8_t speedPin,
    uint8_t steerPin,
    uint8_t relayPin,
    uint8_t estopPin,
    bool revDriveMotor,
    bool revSteerMotor
) {
    // Disengage old relay pin safely if changed
    if (relayPin != _pinRelay) {
        digitalWrite(_pinRelay, LOW);
    }

    _pinSpeed = speedPin;
    _pinSteer = steerPin;
    _pinRelay = relayPin;
    _pinEstop = estopPin;
    _reverseDrive = revDriveMotor;
    _reverseSteer = revSteerMotor;

    pinMode(_pinSpeed, OUTPUT);
    pinMode(_pinSteer, OUTPUT);
    pinMode(_pinRelay, OUTPUT);
    pinMode(_pinEstop, INPUT_PULLUP);

    _recomputeDacTargets(_targetSpeedPct, _targetSteerDeg);
    _applyOutputs();
}

void DacMotorController::setCommand(DriveMode mode, int16_t speedPct, int16_t steerDeg, bool swEstop) {
    _swEstop = swEstop;

    if (_swEstop || mode == MODE_STOP) {
        _mode = MODE_STOP;
        _activeTest = TEST_NONE;
        _targetSpeedPct = 0;
        _targetSteerDeg = 0;
        emergencyNeutral();
        return;
    }

    if (mode != MODE_TEST) {
        _activeTest = TEST_NONE;
    }

    _mode = mode;
    _targetSpeedPct = constrain(speedPct, -100, 100);
    _targetSteerDeg = constrain(steerDeg, -75, 75);

    if (_mode != MODE_TEST || _activeTest == TEST_NONE) {
        _recomputeDacTargets(_targetSpeedPct, _targetSteerDeg);
    }
}

void DacMotorController::triggerTest(TestAction action) {
    if (_swEstop || _hwEstop) {
        emergencyNeutral();
        return;
    }
    _mode = MODE_TEST;
    _activeTest = action;
    _testStartMs = millis();
}

void DacMotorController::_updateTestSequence() {
    if (_mode != MODE_TEST || _activeTest == TEST_NONE) {
        return;
    }

    unsigned long elapsed = millis() - _testStartMs;
    const int16_t TEST_SPEED_PCT = 30; // Safe 30% low-speed verification pulse
    const int16_t TEST_STEER_DEG = 35; // Safe 35-deg steering verification pulse

    switch (_activeTest) {
        case TEST_FWD:
            if (elapsed < 1500) {
                _targetSpeedPct = TEST_SPEED_PCT;
                _targetSteerDeg = 0;
            } else {
                _targetSpeedPct = 0;
                _targetSteerDeg = 0;
                _activeTest = TEST_NONE;
            }
            break;

        case TEST_BACK:
            if (elapsed < 1500) {
                _targetSpeedPct = -TEST_SPEED_PCT;
                _targetSteerDeg = 0;
            } else {
                _targetSpeedPct = 0;
                _targetSteerDeg = 0;
                _activeTest = TEST_NONE;
            }
            break;

        case TEST_LEFT:
            if (elapsed < 1500) {
                _targetSpeedPct = 0;
                _targetSteerDeg = -TEST_STEER_DEG;
            } else {
                _targetSpeedPct = 0;
                _targetSteerDeg = 0;
                _activeTest = TEST_NONE;
            }
            break;

        case TEST_RIGHT:
            if (elapsed < 1500) {
                _targetSpeedPct = 0;
                _targetSteerDeg = TEST_STEER_DEG;
            } else {
                _targetSpeedPct = 0;
                _targetSteerDeg = 0;
                _activeTest = TEST_NONE;
            }
            break;

        case TEST_NEUTRAL:
            _targetSpeedPct = 0;
            _targetSteerDeg = 0;
            if (elapsed >= 1500) {
                _activeTest = TEST_NONE;
            }
            break;

        case TEST_SWEEP:
            if (elapsed < 1200) {
                _targetSpeedPct = TEST_SPEED_PCT;
                _targetSteerDeg = 0;
            } else if (elapsed < 1600) {
                _targetSpeedPct = 0;
                _targetSteerDeg = 0;
            } else if (elapsed < 2800) {
                _targetSpeedPct = -TEST_SPEED_PCT;
                _targetSteerDeg = 0;
            } else if (elapsed < 3200) {
                _targetSpeedPct = 0;
                _targetSteerDeg = 0;
            } else if (elapsed < 4400) {
                _targetSpeedPct = 0;
                _targetSteerDeg = -TEST_STEER_DEG;
            } else if (elapsed < 4800) {
                _targetSpeedPct = 0;
                _targetSteerDeg = 0;
            } else if (elapsed < 6000) {
                _targetSpeedPct = 0;
                _targetSteerDeg = TEST_STEER_DEG;
            } else {
                _targetSpeedPct = 0;
                _targetSteerDeg = 0;
                _activeTest = TEST_NONE;
            }
            break;

        default:
            _targetSpeedPct = 0;
            _targetSteerDeg = 0;
            _activeTest = TEST_NONE;
            break;
    }

    _recomputeDacTargets(_targetSpeedPct, _targetSteerDeg);
}

void DacMotorController::update() {
    _hwEstop = false;

    if (_swEstop || _hwEstop || _mode == MODE_STOP) {
        emergencyNeutral();
        return;
    }

    if (_mode == MODE_TEST) {
        _updateTestSequence();
    }

    _relayEngaged = true;

    unsigned long now = millis();
    if (now - _lastRampMs >= RAMP_INTERVAL_MS) {
        _lastRampMs = now;

        if (_currentDacSpeed < _targetDacSpeed) {
            uint16_t diff = _targetDacSpeed - _currentDacSpeed;
            _currentDacSpeed += (diff > RAMP_MAX_STEP_DAC) ? RAMP_MAX_STEP_DAC : diff;
        } else if (_currentDacSpeed > _targetDacSpeed) {
            uint16_t diff = _currentDacSpeed - _targetDacSpeed;
            _currentDacSpeed -= (diff > RAMP_MAX_STEP_DAC) ? RAMP_MAX_STEP_DAC : diff;
        }

        if (_currentDacSteer < _targetDacSteer) {
            uint16_t diff = _targetDacSteer - _currentDacSteer;
            _currentDacSteer += (diff > RAMP_MAX_STEP_DAC) ? RAMP_MAX_STEP_DAC : diff;
        } else if (_currentDacSteer > _targetDacSteer) {
            uint16_t diff = _currentDacSteer - _targetDacSteer;
            _currentDacSteer -= (diff > RAMP_MAX_STEP_DAC) ? RAMP_MAX_STEP_DAC : diff;
        }

        _applyOutputs();
    }
}

void DacMotorController::emergencyNeutral() {
    _relayEngaged = false;
    _targetSpeedPct = 0;
    _targetSteerDeg = 0;
    _targetDacSpeed = DAC_NEUTRAL;
    _targetDacSteer = DAC_NEUTRAL;
    _currentDacSpeed = DAC_NEUTRAL;
    _currentDacSteer = DAC_NEUTRAL;
    _applyOutputs();
}

void DacMotorController::_recomputeDacTargets(int16_t speedPct, int16_t steerDeg) {
    int16_t effectiveSpeed = _reverseDrive ? -speedPct : speedPct;
    int16_t effectiveSteer = _reverseSteer ? -steerDeg : steerDeg;

    int32_t speedOffset = ((int32_t)effectiveSpeed * DAC_MAX_SWING) / 100;
    _targetDacSpeed = (uint16_t)constrain(DAC_NEUTRAL + speedOffset, DAC_MIN_SAFE, DAC_MAX_SAFE);

    int32_t steerOffset = ((int32_t)effectiveSteer * DAC_MAX_SWING) / 75;
    _targetDacSteer = (uint16_t)constrain(DAC_NEUTRAL + steerOffset, DAC_MIN_SAFE, DAC_MAX_SAFE);
}

void DacMotorController::_writeDacChannel(uint8_t i2cAddr, uint16_t value12bit) {
    uint16_t clamped = constrain(value12bit, 0, DAC_RESOLUTION_MAX);
    uint8_t msb = (uint8_t)((clamped >> 8) & 0x0F);
    uint8_t lsb = (uint8_t)(clamped & 0xFF);

    // Try primary address (0x60 or 0x61 for MCP4725A0)
    Wire.beginTransmission(i2cAddr);
    Wire.write(msb);
    Wire.write(lsb);
    uint8_t err = Wire.endTransmission();

    // Automatic fallback to 0x62 or 0x63 if the red board uses MCP4725A1 chip batch
    if (err != 0) {
        Wire.beginTransmission(i2cAddr + 2);
        Wire.write(msb);
        Wire.write(lsb);
        Wire.endTransmission();
    }
}

void DacMotorController::_applyOutputs() {
    _writeDacChannel(DAC_I2C_ADDR_SPEED, _currentDacSpeed);
    _writeDacChannel(DAC_I2C_ADDR_STEER, _currentDacSteer);

    uint8_t pwmSpeed = (uint8_t)map(_currentDacSpeed, 0, 4095, 0, 255);
    uint8_t pwmSteer = (uint8_t)map(_currentDacSteer, 0, 4095, 0, 255);
    analogWrite(_pinSpeed, pwmSpeed);
    analogWrite(_pinSteer, pwmSteer);

    digitalWrite(_pinRelay, _relayEngaged ? HIGH : LOW);
}
