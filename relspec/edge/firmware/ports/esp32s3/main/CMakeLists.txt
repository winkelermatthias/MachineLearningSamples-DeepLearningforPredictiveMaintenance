idf_component_register(
    SRCS "app_main.c"
         "../../../core/rs_proto.c"
         "../../../core/rs_ring.c"
         "../../../core/rs_sha256.c"
    INCLUDE_DIRS "../../../core"
    REQUIRES esp_wifi esp_event esp_timer nvs_flash lwip)
