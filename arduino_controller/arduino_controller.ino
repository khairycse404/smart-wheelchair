/**
 * ============================================================================
 * SMART WHEELCHAIR - MODULAR ARDUINO MOTOR, DAC & GPS CONTROLLER
 * Mechatronics Engineering Graduation Project
 * ============================================================================
 * Architecture:
 *   [HP Mini PC (Python Serial Bridge @ 115200 baud)]
 *          |
 *          v
 *   [Arduino Uno / Nano / Mega]
 *     |-- Module 1: DacMotorController (DacMotorController.h / .cpp)
 *     |     |-- Dual MCP4725 12-Bit I2C DACs (0x60 Speed Y-Axis, 0x61 Steer X-Axis)
 *     |     |-- Runtime-Customizable PWM & Takeover Relay Pins from Web GUI
 *     |     |-- Runtime-Customizable Motor Direction Reversal Flags
 *     |     |-- Safe 2.50V Center Neutral Initialization + Smooth Acceleration Ramp
 *     |     |-- Pre-Autonomous Motor Test Suite (Forward, Reverse, Left, Right, Sweep)
 *     |
 *     |-- Module 2: GpsModule (GpsModule.h / .cpp)
 *           |-- Ai-Thinker GP-01-Kit UART Receiver (Configurable RX/TX Pins)
 *           |-- Live ON/OFF Toggle from Web GUI
 *           |-- Clearly Marked Manual Integration Placeholders for Future Expansion
 *
 * Serial Protocol (115200 baud, newline '\n' terminated):
 *   Incoming from Mini PC:
 *     CMD:<mode>,<speed_pct>,<steer_deg>,<estop>
 *       - mode: STOP (0), MANUAL (1), TEST (2), AUTO (3)
 *       - speed_pct: -100 to +100
 *       - steer_deg: -75 to +75
 *       - estop: 0 or 1
 *     CFG:<pin_spd>,<pin_str>,<pin_rly>,<pin_estp>,<pin_gps_rx>,<pin_gps_tx>,<rev_drive>,<rev_steer>
 *     GPS:<0|1>
 *     TEST:<FWD|BACK|LEFT|RIGHT|NEUTRAL|SWEEP>
 *
 *   Outgoing to Mini PC (10 Hz):
 *     STAT:MODE=...,SPD=...,STR=...,DAC_Y=...,DAC_X=...,V_Y=...,V_X=...,RLY=...,ESTOP=...,PINS=...,REV=...,GPS_EN=...,GPS_ST=...,GPS_FIX=...,SATS=...,LAT=...,LON=...
 * ============================================================================
 */

#include <Arduino.h>
#include <Wire.h>
#include "Config.h"
#include "DacMotorController.h"
#include "GpsModule.h"

DacMotorController motorController;
GpsModule gpsModule;

char serialRxBuf[128];
uint8_t serialRxIdx = 0;
unsigned long lastCommandRxMs = 0;
unsigned long lastTelemetryTxMs = 0;

static DriveMode parseModeToken(const char* tok) {
    if (strcmp(tok, "AUTO") == 0 || strcmp(tok, "3") == 0) return MODE_AUTONOMOUS;
    if (strcmp(tok, "MANUAL") == 0 || strcmp(tok, "1") == 0) return MODE_MANUAL;
    if (strcmp(tok, "TEST") == 0 || strcmp(tok, "2") == 0) return MODE_TEST;
    return MODE_STOP;
}

static const char* modeToString(DriveMode m) {
    switch (m) {
        case MODE_MANUAL:     return "MANUAL";
        case MODE_TEST:       return "TEST";
        case MODE_AUTONOMOUS: return "AUTO";
        default:              return "STOP";
    }
}

void processSerialCommand(char* line) {
    lastCommandRxMs = millis();

    // 1. CMD:<mode>,<speed_pct>,<steer_deg>,<estop>
    if (strncmp(line, "CMD:", 4) == 0) {
        char* p = line + 4;
        char* tokMode  = strtok(p, ",");
        char* tokSpeed = strtok(nullptr, ",");
        char* tokSteer = strtok(nullptr, ",");
        char* tokEstop = strtok(nullptr, ",");

        if (tokMode && tokSpeed && tokSteer && tokEstop) {
            DriveMode mode = parseModeToken(tokMode);
            int16_t spd = (int16_t)atoi(tokSpeed);
            int16_t str = (int16_t)atoi(tokSteer);
            bool estop  = (atoi(tokEstop) != 0);
            motorController.setCommand(mode, spd, str, estop);
        }
    }
    // 2. CFG:<pin_spd>,<pin_str>,<pin_rly>,<pin_estp>,<pin_gps_rx>,<pin_gps_tx>,<rev_drive>,<rev_steer>
    else if (strncmp(line, "CFG:", 4) == 0) {
        char* p = line + 4;
        char* tSpd   = strtok(p, ",");
        char* tStr   = strtok(nullptr, ",");
        char* tRly   = strtok(nullptr, ",");
        char* tEstp  = strtok(nullptr, ",");
        char* tGpsRx = strtok(nullptr, ",");
        char* tGpsTx = strtok(nullptr, ",");
        char* tRevD  = strtok(nullptr, ",");
        char* tRevS  = strtok(nullptr, ",");

        if (tSpd && tStr && tRly && tEstp && tGpsRx && tGpsTx && tRevD && tRevS) {
            motorController.configureHardware(
                (uint8_t)atoi(tSpd),
                (uint8_t)atoi(tStr),
                (uint8_t)atoi(tRly),
                (uint8_t)atoi(tEstp),
                (atoi(tRevD) != 0),
                (atoi(tRevS) != 0)
            );
            gpsModule.configurePins((uint8_t)atoi(tGpsRx), (uint8_t)atoi(tGpsTx));
        }
    }
    // 3. GPS:<0|1>
    else if (strncmp(line, "GPS:", 4) == 0) {
        bool en = (atoi(line + 4) != 0);
        gpsModule.setEnabled(en);
    }
    // 4. TEST:<FWD|BACK|LEFT|RIGHT|NEUTRAL|SWEEP>
    else if (strncmp(line, "TEST:", 5) == 0) {
        const char* act = line + 5;
        if (strcmp(act, "FWD") == 0)          motorController.triggerTest(TEST_FWD);
        else if (strcmp(act, "BACK") == 0)    motorController.triggerTest(TEST_BACK);
        else if (strcmp(act, "LEFT") == 0)    motorController.triggerTest(TEST_LEFT);
        else if (strcmp(act, "RIGHT") == 0)   motorController.triggerTest(TEST_RIGHT);
        else if (strcmp(act, "NEUTRAL") == 0) motorController.triggerTest(TEST_NEUTRAL);
        else if (strcmp(act, "SWEEP") == 0)   motorController.triggerTest(TEST_SWEEP);
        else                                  motorController.triggerTest(TEST_NONE);
    }
}

void publishTelemetry() {
    Serial.print(F("STAT:MODE="));
    Serial.print(modeToString(motorController.getMode()));
    Serial.print(F(",SPD="));
    Serial.print(motorController.getCommandSpeedPct());
    Serial.print(F(",STR="));
    Serial.print(motorController.getCommandSteerDeg());
    Serial.print(F(",DAC_Y="));
    Serial.print(motorController.getCurrentDacSpeed());
    Serial.print(F(",DAC_X="));
    Serial.print(motorController.getCurrentDacSteer());
    Serial.print(F(",V_Y="));
    Serial.print(motorController.getVoltageSpeed(), 2);
    Serial.print(F(",V_X="));
    Serial.print(motorController.getVoltageSteer(), 2);
    Serial.print(F(",RLY="));
    Serial.print(motorController.isRelayEngaged() ? 1 : 0);
    Serial.print(F(",ESTOP="));
    Serial.print(motorController.isEstopActive() ? 1 : 0);
    Serial.print(F(",PINS="));
    Serial.print(motorController.getPinSpeed());
    Serial.print('/');
    Serial.print(motorController.getPinSteer());
    Serial.print('/');
    Serial.print(motorController.getPinRelay());
    Serial.print('/');
    Serial.print(motorController.getPinEstop());
    Serial.print('/');
    Serial.print(gpsModule.getRxPin());
    Serial.print('/');
    Serial.print(gpsModule.getTxPin());
    Serial.print(F(",REV="));
    Serial.print(motorController.isDriveReversed() ? 1 : 0);
    Serial.print('/');
    Serial.print(motorController.isSteerReversed() ? 1 : 0);
    Serial.print(F(",GPS_EN="));
    Serial.print(gpsModule.isEnabled() ? 1 : 0);
    Serial.print(F(",GPS_ST="));
    Serial.print(gpsModule.getStatusText());
    Serial.print(F(",GPS_FIX="));
    Serial.print(gpsModule.hasFix() ? 1 : 0);
    Serial.print(F(",SATS="));
    Serial.print(gpsModule.getSatellites());
    Serial.print(F(",LAT="));
    Serial.print(gpsModule.getLatitude(), 6);
    Serial.print(F(",LON="));
    Serial.println(gpsModule.getLongitude(), 6);
}

void setup() {
    // 1. Initialize DAC & Motor outputs to safe 2.50V neutral immediately
    motorController.begin();

    // 2. Initialize GPS Module (starts DISABLED until toggled ON from GUI)
    gpsModule.begin(PIN_GPS_RX_DEFAULT, PIN_GPS_TX_DEFAULT);

    // 3. Start high-speed Serial link to HP Mini PC
    Serial.begin(SERIAL_BAUD_RATE);
    lastCommandRxMs = millis();
}

void loop() {
    // 1. Read incoming serial commands from Mini PC
    while (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == '\r') continue;
        if (c == '\n') {
            serialRxBuf[serialRxIdx] = '\0';
            if (serialRxIdx > 0) {
                processSerialCommand(serialRxBuf);
            }
            serialRxIdx = 0;
        } else if (serialRxIdx < sizeof(serialRxBuf) - 1) {
            serialRxBuf[serialRxIdx++] = c;
        } else {
            serialRxIdx = 0;
        }
    }

    // 2. Safety Watchdog: If Mini PC disconnects or stops polling > 1.2s, drop to neutral
    if (motorController.getMode() != MODE_STOP &&
        motorController.getMode() != MODE_TEST &&
        (millis() - lastCommandRxMs > COMMAND_WATCHDOG_MS)) {
        motorController.setCommand(MODE_STOP, 0, 0, false);
    }

    // 3. Update DAC motor controller ramping & test timers
    motorController.update();

    // 4. Update GPS module (only processes UART bytes if enabled from GUI)
    gpsModule.update();

    // 5. Stream live status back to Mini PC GUI at 10 Hz
    if (millis() - lastTelemetryTxMs >= TELEMETRY_INTERVAL_MS) {
        lastTelemetryTxMs = millis();
        publishTelemetry();
    }
}
