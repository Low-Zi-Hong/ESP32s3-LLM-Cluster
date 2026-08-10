idf_component_register(SRCS "embedding.cpp" "main.cpp"
                       REQUIRES spi_flash esp_psram esp_partition esp_timer esp_common
                       INCLUDE_DIRS ".")