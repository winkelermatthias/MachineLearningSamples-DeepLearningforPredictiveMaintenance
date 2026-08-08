/* sim_main.c - drives the real firmware over a recorded scenario.
 *
 * in.bin  : [uint32 n_frames][uint32 n_samples][float32 fs]
 *           then per frame: [uint32 t_s][uint8 truth_running][float32 x[n]]
 * cfg.txt : optional "t_s key value" lines applied as cloud pushes
 * out.csv : one row per wake cycle
 */
#include "edge_policy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 32768
static float xbuf[MAXN];

typedef struct { uint32_t t; char key[40]; double val; } cmd_t;
static cmd_t cmds[256]; static int ncmd = 0, cmdi = 0;

static void load_cmds(const char *p)
{
    FILE *f = fopen(p, "r"); if (!f) return;
    char k[40]; unsigned t; double v;
    while (ncmd < 256 && fscanf(f, "%u %39s %lf", &t, k, &v) == 3) {
        cmds[ncmd].t = t; strncpy(cmds[ncmd].key, k, 39); cmds[ncmd].val = v; ncmd++;
    }
    fclose(f);
}

static void apply_key(ep_config_t *c, const char *k, double v)
{
    if      (!strcmp(k,"wake_interval_s"))       c->wake_interval_s = (uint32_t)v;
    else if (!strcmp(k,"heartbeat_s"))           c->heartbeat_s = (uint32_t)v;
    else if (!strcmp(k,"upload_max_per_window")) c->upload_max_per_window = (uint16_t)v;
    else if (!strcmp(k,"upload_reserve_change")) c->upload_reserve_change = (uint8_t)v;
    else if (!strcmp(k,"conn_max_per_day"))      c->conn_max_per_day = (uint8_t)v;
    else if (!strcmp(k,"conn_priority_extra"))   c->conn_priority_extra = (uint8_t)v;
    else if (!strcmp(k,"state_persist_n"))       c->state_persist_n = (uint8_t)v;
    else if (!strcmp(k,"state_hyst_db"))         c->state_hyst_db = (float)v;
    else if (!strcmp(k,"batch_max_frames"))      c->batch_max_frames = (uint8_t)v;
    else if (!strcmp(k,"batch_max_bytes"))       c->batch_max_bytes = (uint16_t)v;
    else if (!strcmp(k,"batch_max_age_s"))       c->batch_max_age_s = (uint32_t)v;
    else if (!strcmp(k,"conn_interval_s"))       c->conn_interval_s = (uint32_t)v;
    else if (!strcmp(k,"conn_burst_max"))        c->conn_burst_max = (uint8_t)v;
    else if (!strcmp(k,"conn_burst_window_s"))   c->conn_burst_window_s = (uint32_t)v;
    else if (!strcmp(k,"drift_breaks_schedule")) c->drift_breaks_schedule = (uint8_t)v;
    else if (!strcmp(k,"step_min"))              c->step_min = (float)v;
    else if (!strcmp(k,"flush_on_change"))       c->flush_on_change = (uint8_t)v;
    else if (!strcmp(k,"s_hi"))                  c->s_hi = (float)v;
    else if (!strcmp(k,"s_lo"))                  c->s_lo = (float)v;
    else if (!strcmp(k,"persist_n"))             c->persist_n = (uint8_t)v;
    else if (!strcmp(k,"refractory_s"))          c->refractory_s = (uint32_t)v;
    else if (!strcmp(k,"change_burst_max"))      c->change_burst_max = (uint8_t)v;
    else if (!strcmp(k,"backoff_mult"))          c->backoff_mult = (uint8_t)v;
    else if (!strcmp(k,"cusum_h"))               c->cusum_h = (float)v;
    else if (!strcmp(k,"cusum_k"))               c->cusum_k = (float)v;
    else if (!strcmp(k,"slow_freeze_n"))         c->slow_freeze_n = (uint32_t)v;
    else if (!strcmp(k,"min_vel_delta"))         c->min_vel_delta = (float)v;
    else if (!strcmp(k,"min_acc_delta_db"))      c->min_acc_delta_db = (float)v;
    else if (!strcmp(k,"idle_gating_enable"))    c->idle_gating_enable = (uint8_t)v;
    else if (!strcmp(k,"discovery_min_n"))       c->discovery_min_n = (uint32_t)v;
    else if (!strcmp(k,"discovery_min_sep"))     c->discovery_min_sep = (float)v;
    else if (!strcmp(k,"discovery_min_gap_db"))  c->discovery_min_gap_db = (float)v;
    else if (!strcmp(k,"cmd_reset_baseline"))    c->cmd_reset_baseline = (uint8_t)v;
    else if (!strcmp(k,"cmd_reset_discovery"))   c->cmd_reset_discovery = (uint8_t)v;
    else if (!strcmp(k,"cmd_force_anchor"))      c->cmd_force_anchor = (uint8_t)v;
    else if (!strcmp(k,"cmd_force_upload"))      c->cmd_force_upload = (uint8_t)v;
    else fprintf(stderr, "unknown key %s\n", k);
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr,"usage: sim in.bin out.csv [cfg.txt] [payloads.bin]\n"); return 1; }
    FILE *in = fopen(argv[1],"rb"); if(!in){perror("in");return 1;}
    FILE *out = fopen(argv[2],"w"); if(!out){perror("out");return 1;}
    if (argc > 3 && strcmp(argv[3],"-")) load_cmds(argv[3]);
    FILE *pl = (argc > 4) ? fopen(argv[4],"wb") : NULL;

    uint32_t nf, ns; float fs;
    if (fread(&nf,4,1,in)!=1 || fread(&ns,4,1,in)!=1 || fread(&fs,4,1,in)!=1) return 1;
    if (ns > MAXN) { fprintf(stderr,"n too large\n"); return 1; }

    ep_config_t cfg; ep_defaults(&cfg);
    ep_ctx_t ctx; ep_init(&ctx, &cfg);

    fprintf(out,"i,t_s,day,truth_running,state,disc_valid,disc_thr_db,disc_sep,"
        "disc_mu_idle,disc_mu_run,decision,payload,queued,connected,conn_bytes,"
        "conn_frames,kind,jump,score,acc_rms,vel_rms,acc_kurt,acc_crest,env_rms,"
        "env_kurt,env_crest,fr_hz,fr_conf,tier,mask_bands,mask_max_db,base_n,"
        "abs_ok,drift,drift_feat");
    for (int b=0;b<EG_NBANDS;b++) fprintf(out,",bd%d",b);
    for (int o=0;o<EG_NORDERS;o++) fprintf(out,",od%d",o);
    fprintf(out,"\n");

    for (uint32_t i=0;i<nf;i++) {
        uint32_t t; uint8_t truth;
        if (fread(&t,4,1,in)!=1) break;
        if (fread(&truth,1,1,in)!=1) break;
        if (fread(xbuf,4,ns,in)!=ns) break;

        while (cmdi < ncmd && cmds[cmdi].t <= t) {
            ep_config_t nc = ctx.cfg;
            /* a push carries one or more keys sharing the same timestamp */
            uint32_t tt = cmds[cmdi].t;
            while (cmdi < ncmd && cmds[cmdi].t == tt) {
                apply_key(&nc, cmds[cmdi].key, cmds[cmdi].val); cmdi++;
            }
            nc.version = ctx.applied_version + 1;
            ep_apply_config(&ctx, &nc, t);
        }

        eg_features_t f; eg_change_t g; memset(&g,0,sizeof(g));
        ep_result_t r = ep_wake(&ctx, xbuf, ns, t, &f, &g);
        if (pl && r.payload_len) {
            /* record index, timestamp, estimated speed, then the raw payload,
             * plus the quantised truth so the decoder can be scored */
            uint32_t L = r.payload_len;
            fwrite(&i,4,1,pl); fwrite(&t,4,1,pl); fwrite(&f.fr_hz,4,1,pl);
            fwrite(&L,4,1,pl); fwrite(eg_last_payload(),1,L,pl);
            fwrite(eg_last_u8(),1,EG_NBINS,pl);
        }

        fprintf(out,"%u,%u,%.3f,%u,%s,%d,%.2f,%.3f,%.2f,%.2f,%s,%u,%u,%u,%u,%u,"
                    "%u,%.2f,%.3f,%.6f,%.4f,%.3f,%.3f,%.6f,%.3f,%.3f,%.3f,%.2f,%u,%u,%.2f,%u,%u,%.2f,%u",
            i, t, t/86400.0, truth, ep_state_name(r.state),
            ctx.disc.valid?1:0, ctx.disc.thr_db, ctx.disc.sep,
            ctx.disc.mu_idle_db, ctx.disc.mu_run_db,
            eg_decision_name(r.decision), r.payload_len, r.queued, r.connected,
            r.conn_bytes, r.conn_frames, r.change_kind, g.jump, g.score,
            f.acc_rms, f.vel_rms, f.acc_kurt, f.acc_crest,
            f.env_rms, f.env_kurt, f.env_crest, f.fr_hz, f.fr_conf, f.align_tier,
            g.mask_bands, g.mask_max_db, ctx.eg.base.n, g.abs_floor_ok?1:0, g.drift, g.drift_feat);
        for (int b=0;b<EG_NBANDS;b++) fprintf(out,",%.2f", g.band_delta_db[b]);
        for (int o=0;o<EG_NORDERS;o++) fprintf(out,",%.2f", g.order_delta_db[o]);
        fprintf(out,"\n");
    }
    fprintf(stderr,"acq=%u idle_skipped=%u conns=%u dropped=%u disc_valid=%d "
                   "thr=%.1f sep=%.2f\n",
        ctx.n_acq, ctx.n_idle_skipped, ctx.q.n_conn_total, ctx.q.n_dropped,
        ctx.disc.valid, ctx.disc.thr_db, ctx.disc.sep);
    if (pl) fclose(pl);
    fclose(in); fclose(out); return 0;
}
