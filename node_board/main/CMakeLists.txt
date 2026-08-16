idf_component_register(SRCS "bitlinear_forward.S" "qwen_attention.cpp" "bitlinear.cpp" "spi_bus.cpp" "main.cpp"
                       PRIV_REQUIRES spi_flash esp_psram esp_partition esp_timer esp_common esp_driver_spi esp_driver_gpio
                       INCLUDE_DIRS "")
