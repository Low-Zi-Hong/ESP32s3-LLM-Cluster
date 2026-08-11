idf_component_register(SRCS "lm_head.cpp" "spi_bus.cpp" "embedding.cpp" "main.cpp"
                       REQUIRES spi_flash esp_psram esp_partition esp_timer esp_common esp_driver_spi
                       INCLUDE_DIRS ".")