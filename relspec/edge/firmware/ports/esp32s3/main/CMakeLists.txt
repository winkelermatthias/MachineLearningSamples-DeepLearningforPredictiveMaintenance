idf_component_register(
    SRCS "app_main.c"
         "../../../core/rs_proto.c"
         "../../../core/rs_ring.c"
    INCLUDE_DIRS "../../../core"
    REQUIRES esp_wifi esp_event esp_timer nvs_flash lwip)
