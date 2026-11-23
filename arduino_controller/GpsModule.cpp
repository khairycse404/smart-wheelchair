#include "GpsModule.h"
#include <string.h>
#include <stdlib.h>

GpsModule::GpsModule()
    : _rxPin(PIN_GPS_RX_DEFAULT),
      _txPin(PIN_GPS_TX_DEFAULT),
      _enabled(false),
      _gpsSerial(nullptr),
      _hasFix(false),
      _satellites(0),
      _latitude(0.0f),
      _longitude(0.0f),
      _lineIdx(0) {
    strncpy(_statusText, "DISABLED", sizeof(_statusText) - 1);
    _statusText[sizeof(_statusText) - 1] = '\0';
}

GpsModule::~GpsModule() {
    if (_gpsSerial != nullptr) {
        delete _gpsSerial;
        _gpsSerial = nullptr;
    }
}

void GpsModule::begin(uint8_t rxPin, uint8_t txPin) {
    configurePins(rxPin, txPin);
}

void GpsModule::configurePins(uint8_t rxPin, uint8_t txPin) {
    bool wasEnabled = _enabled;
    if (_gpsSerial != nullptr) {
        _gpsSerial->end();
        delete _gpsSerial;
        _gpsSerial = nullptr;
    }
    _rxPin = rxPin;
    _txPin = txPin;
    _gpsSerial = new SoftwareSerial(_rxPin, _txPin);
    if (wasEnabled) {
        _gpsSerial->begin(GPS_DEFAULT_BAUD_RATE);
    }
}

void GpsModule::setEnabled(bool enabled) {
    if (_enabled == enabled) {
        return;
    }
    _enabled = enabled;
    _lineIdx = 0;

    if (_enabled) {
        if (_gpsSerial == nullptr) {
            _gpsSerial = new SoftwareSerial(_rxPin, _txPin);
        }
        _gpsSerial->begin(GPS_DEFAULT_BAUD_RATE);
        strncpy(_statusText, "SEARCHING_SATS", sizeof(_statusText) - 1);
    } else {
        if (_gpsSerial != nullptr) {
            _gpsSerial->end();
        }
        _hasFix = false;
        strncpy(_statusText, "DISABLED", sizeof(_statusText) - 1);
    }
}

void GpsModule::update() {
    if (!_enabled || _gpsSerial == nullptr) {
        return;
    }

    // ========================================================================
    // [MANUAL GPS INTEGRATION PLACEHOLDER - PART 1: RAW STREAM READING]
    // When GPS is enabled from the GUI, bytes from the Ai-Thinker GP-01-Kit
    // arrive here at 9600 baud. You can replace or extend _parseNmeaLine()
    // below with TinyGPS++ or custom waypoint logic after outdoor testing.
    // ========================================================================
    while (_gpsSerial->available() > 0) {
        char c = (char)_gpsSerial->read();
        if (c == '\r') {
            continue;
        }
        if (c == '\n') {
            _lineBuf[_lineIdx] = '\0';
            if (_lineIdx > 6 && _lineBuf[0] == '$') {
                _parseNmeaLine(_lineBuf);
            }
            _lineIdx = 0;
        } else if (_lineIdx < sizeof(_lineBuf) - 1) {
            _lineBuf[_lineIdx++] = c;
        } else {
            _lineIdx = 0;
        }
    }
}

void GpsModule::_parseNmeaLine(const char* line) {
    // ========================================================================
    // [MANUAL GPS INTEGRATION PLACEHOLDER - PART 2: CUSTOM GPS PARSER]
    // Add your custom NMEA / TinyGPS++ coordinate and waypoint extraction here.
    // Assign:
    //   _hasFix     = true/false;
    //   _latitude   = ...;
    //   _longitude  = ...;
    //   _satellites = ...;
    //   strncpy(_statusText, "3D_FIX_LOCKED", sizeof(_statusText) - 1);
    // ========================================================================

    // Built-in lightweight parser for $GNGGA / $GPGGA (Satellites + Fix + Lat/Lon)
    if (strncmp(line, "$GNGGA,", 7) == 0 || strncmp(line, "$GPGGA,", 7) == 0) {
        char buf[96];
        strncpy(buf, line, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';

        char* fields[15] = {nullptr};
        uint8_t fCount = 0;
        char* p = buf;
        fields[fCount++] = p;
        while (*p && fCount < 15) {
            if (*p == ',') {
                *p = '\0';
                fields[fCount++] = p + 1;
            }
            p++;
        }

        if (fCount >= 8) {
            int fixQual = atoi(fields[6]);
            _satellites = (uint8_t)atoi(fields[7]);
            if (fixQual > 0 && strlen(fields[2]) > 4 && strlen(fields[4]) > 4) {
                _hasFix = true;
                _latitude = _convertNmeaCoord(fields[2], fields[3][0]);
                _longitude = _convertNmeaCoord(fields[4], fields[5][0]);
                strncpy(_statusText, "FIX_LOCKED", sizeof(_statusText) - 1);
            } else {
                _hasFix = false;
                strncpy(_statusText, "NO_FIX_INDOOR", sizeof(_statusText) - 1);
            }
        }
    }
}

float GpsModule::_convertNmeaCoord(const char* valStr, char hemi) {
    float raw = atof(valStr);
    int deg = (int)(raw / 100.0f);
    float minutes = raw - (deg * 100.0f);
    float dec = (float)deg + (minutes / 60.0f);
    if (hemi == 'S' || hemi == 'W') {
        dec = -dec;
    }
    return dec;
}
