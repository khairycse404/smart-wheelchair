#ifndef GPS_MODULE_H
#define GPS_MODULE_H

#include <Arduino.h>
#include <SoftwareSerial.h>
#include "Config.h"

class GpsModule {
public:
    GpsModule();
    ~GpsModule();

    // Initializes GPS module on configurable RX/TX pins
    void begin(uint8_t rxPin = PIN_GPS_RX_DEFAULT, uint8_t txPin = PIN_GPS_TX_DEFAULT);

    // Reconfigures GPS RX/TX pins at runtime from the Web GUI
    void configurePins(uint8_t rxPin, uint8_t txPin);

    // Turns GPS reading ON or OFF from the Web GUI
    void setEnabled(bool enabled);
    bool isEnabled() const { return _enabled; }

    // Non-blocking serial reader & parser (only runs when _enabled == true)
    void update();

    // Telemetry getters for GUI display
    bool hasFix() const { return _hasFix; }
    uint8_t getSatellites() const { return _satellites; }
    float getLatitude() const { return _latitude; }
    float getLongitude() const { return _longitude; }
    const char* getStatusText() const { return _statusText; }
    uint8_t getRxPin() const { return _rxPin; }
    uint8_t getTxPin() const { return _txPin; }

private:
    uint8_t _rxPin;
    uint8_t _txPin;
    bool _enabled;
    SoftwareSerial* _gpsSerial;

    bool _hasFix;
    uint8_t _satellites;
    float _latitude;
    float _longitude;
    char _statusText[24];

    char _lineBuf[96];
    uint8_t _lineIdx;

    void _parseNmeaLine(const char* line);
    float _convertNmeaCoord(const char* valStr, char hemi);
};

#endif // GPS_MODULE_H
