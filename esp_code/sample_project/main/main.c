#include <stdio.h>
#include <esp_wifi_types.h>
#include "esp_wifi.h"
#include "esp_err.h"
#include "esp_camera.h"
#include "nvs_flash.h"

// Wifi
#define ESP_MAXIMUM_RETRY  5
#define WIFI_SSID "eir82458730"
#define WIFI_PASSWORD "19d433638b6b"

// Camera
#define BOARD_ESP32CAM_AITHINKER

// Camera pins
#define CAM_PIN_PWDN 32
#define CAM_PIN_RESET -1 //software reset will be performed
#define CAM_PIN_XCLK 0
#define CAM_PIN_SIOD 26
#define CAM_PIN_SIOC 27
#define CAM_PIN_D7 35
#define CAM_PIN_D6 34
#define CAM_PIN_D5 39
#define CAM_PIN_D4 36
#define CAM_PIN_D3 21
#define CAM_PIN_D2 19
#define CAM_PIN_D1 18
#define CAM_PIN_D0 5
#define CAM_PIN_VSYNC 25
#define CAM_PIN_HREF 23
#define CAM_PIN_PCLK 22

// Camera Config
static camera_config_t camera_config = {
    .pin_pwdn = CAM_PIN_PWDN,
    .pin_reset = CAM_PIN_RESET,
    .pin_xclk = CAM_PIN_XCLK,
    .pin_sscb_sda = CAM_PIN_SIOD,
    .pin_sscb_scl = CAM_PIN_SIOC,

    .pin_d7 = CAM_PIN_D7,
    .pin_d6 = CAM_PIN_D6,
    .pin_d5 = CAM_PIN_D5,
    .pin_d4 = CAM_PIN_D4,
    .pin_d3 = CAM_PIN_D3,
    .pin_d2 = CAM_PIN_D2,
    .pin_d1 = CAM_PIN_D1,
    .pin_d0 = CAM_PIN_D0,
    .pin_vsync = CAM_PIN_VSYNC,
    .pin_href = CAM_PIN_HREF,
    .pin_pclk = CAM_PIN_PCLK,

    //XCLK 20MHz or 10MHz for OV2640 double FPS (Experimental)
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_JPEG, //YUV422,GRAYSCALE,RGB565,JPEG
    .frame_size = FRAMESIZE_VGA,    //QQVGA-UXGA Do not use sizes above QVGA when not JPEG

    .jpeg_quality = 12, //0-63 lower number means higher quality
    .fb_count = 1       //if more than one, i2s runs in continuous mode. Use only with JPEG
};

static esp_err_t init_camera()
{
    //initialize the camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK)
    {
        printf("Camera Init Failed");
        return err;
    }

    return ESP_OK;
}

void wifi_init_sta(void){

    // https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi.html#esp32-wi-fi-station-general-scenario 
    
    nvs_flash_init();   //Initialize NVS

    ESP_ERROR_CHECK(esp_netif_init()); // Creating a LwIP core task and initialize LwIP-related work 
    ESP_ERROR_CHECK(esp_event_loop_create_default()); // Creating a system Event task and initializing an application event’s callback function
    
    esp_netif_create_default_wifi_sta(); // Creating default network interface instance binding station with TCP/IP stack
    wifi_init_config_t config = WIFI_INIT_CONFIG_DEFAULT(); // initialize the config to default values. https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_wifi.html#_CPPv418wifi_init_config_t
    ESP_ERROR_CHECK(esp_wifi_init(&config));  // Creating Wi-Fi driver task and initializing Wi-Fi driver
    printf("\n\n\rwifi station initialized.\n\n\r");

    /* if the configuration does not need to change after the Wi-Fi connection is set up
       you should configure the Wi-Fi driver at this stage */

    wifi_sta_config_t wifi_config = {
        .ssid = WIFI_SSID,
        .password = WIFI_PASSWORD,
        .scan_method = WIFI_FAST_SCAN,
        .threshold.authmode = WIFI_AUTH_WPA2_PSK
    };
    printf("\n\n\rwifi station configured.\n\n\r");

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA)); //configuring the Wi-Fi mode as Station
    ESP_ERROR_CHECK(esp_wifi_set_config(ESP_IF_WIFI_STA, &wifi_config));
}



void app_main(void)
{
    uint8_t count = 0;

    wifi_init_sta();

    ESP_ERROR_CHECK(esp_wifi_start());
    printf("\n\n\rwifi station started.\n\n\r");
    ESP_ERROR_CHECK(esp_wifi_connect());
    printf("\n\n\rwifi station connected.\n\n\r");

    init_camera();

    while (count < 5)
    {
        printf("\n\n\rTaking picture...\n\r");
        camera_fb_t *pic = esp_camera_fb_get();

        // use pic->buf to access the image
        printf("Picture taken! Its size was: %zu bytes\n\n\r", pic->len);
        count++;
        vTaskDelay(5000 / portTICK_RATE_MS);

    }

    ESP_ERROR_CHECK(esp_wifi_disconnect());
    printf("\n\n\rwifi station disconnected.\n\n\r");
    ESP_ERROR_CHECK(esp_wifi_stop());
    printf("\n\n\rwifi station stopped.\n\n\r");
    ESP_ERROR_CHECK(esp_wifi_deinit());
    printf("\n\n\rwifi station deinitialized.\n\n\r");



}

