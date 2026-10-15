/* Only this bridge knows MuJoCo's versioned C structs. Rust owns an opaque handle.
 * Resolve the C API dynamically: the Python wheel ships no Windows import lib.
 * Neither this bridge nor the executable invokes or embeds Python. */
#include <mujoco/mujoco.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if mjVERSION_HEADER != 3014000
#error "This bridge is pinned to MuJoCo 3.14.0 headers"
#endif

typedef struct {
  void* library;
  mjModel* model;
  mjData* data;
  int q[5], v[5], a[5];
  int pairs[64][2], pair_count;
  double lower[5], upper[5], pending;
  int (*version)(void);
  mjModel* (*load)(const char*, const mjVFS*, char*, int);
  mjData* (*make)(const mjModel*);
  void (*delete_model)(mjModel*);
  void (*delete_data)(mjData*);
  void (*reset)(const mjModel*, mjData*);
  void (*forward)(const mjModel*, mjData*);
  void (*step)(const mjModel*, mjData*);
  int (*name2id)(const mjModel*, int, const char*);
  const char* (*id2name)(const mjModel*, int, int);
  void (*contact_force)(const mjModel*, const mjData*, int, mjtNum*);
} LampPhysics;

typedef struct {
  int contacts, warnings;
  double penetration_m, contact_force_n, max_effort_nm;
} LampDiagnostics;

static void* symbol(void* library, const char* name) {
#ifdef _WIN32
  return (void*)GetProcAddress((HMODULE)library, name);
#else
  return dlsym(library, name);
#endif
}

void lamp_close(LampPhysics* p) {
  if (!p) return;
  if (p->data && p->delete_data) p->delete_data(p->data);
  if (p->model && p->delete_model) p->delete_model(p->model);
  if (p->library) {
#ifdef _WIN32
    FreeLibrary((HMODULE)p->library);
#else
    dlclose(p->library);
#endif
  }
  free(p);
}

LampPhysics* lamp_open(const char* library, const char* xml, const char* const* names,
                       const double* lower, const double* upper, char* error, int error_size) {
  LampPhysics* p = (LampPhysics*)calloc(1, sizeof(LampPhysics));
  if (!p) { snprintf(error, error_size, "Cannot allocate physics handle"); return NULL; }
#ifdef _WIN32
  /* Convert UTF-8 paths to Windows UTF-16 without modifying the process DLL search path. */
  int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, library, -1, NULL, 0);
  wchar_t* wide = length > 0 ? (wchar_t*)malloc(length * sizeof(wchar_t)) : NULL;
  if (wide) { MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, library, -1, wide, length); p->library = (void*)LoadLibraryW(wide); free(wide); }
#else
  p->library = dlopen(library, RTLD_NOW | RTLD_LOCAL);
#endif
  if (!p->library) { snprintf(error, error_size, "Cannot load MuJoCo shared library: %s", library); lamp_close(p); return NULL; }
#define LOAD(field, api) do { void* address = symbol(p->library, api); if (!address) { snprintf(error, error_size, "Missing C API: %s", api); lamp_close(p); return NULL; } memcpy(&p->field, &address, sizeof(address)); } while (0)
  LOAD(version, "mj_version");
  if (p->version() != mjVERSION_HEADER) { snprintf(error, error_size, "MuJoCo version mismatch: runtime=%d, headers=%d", p->version(), mjVERSION_HEADER); lamp_close(p); return NULL; }
  LOAD(load, "mj_loadXML"); LOAD(make, "mj_makeData");
  LOAD(delete_model, "mj_deleteModel"); LOAD(delete_data, "mj_deleteData");
  LOAD(reset, "mj_resetData"); LOAD(forward, "mj_forward"); LOAD(step, "mj_step");
  LOAD(name2id, "mj_name2id"); LOAD(id2name, "mj_id2name"); LOAD(contact_force, "mj_contactForce");
#undef LOAD
  p->model = p->load(xml, NULL, error, error_size);
  if (!p->model) { lamp_close(p); return NULL; }
  if (p->model->nq != 5 || p->model->nv != 5 || p->model->nu != 5 || fabs(p->model->opt.timestep - 1.0/240.0) > 1e-12) {
    snprintf(error, error_size, "Expected five scalar joints/actuators and a 240 Hz timestep"); lamp_close(p); return NULL;
  }
  for (int i = 0; i < 5; i++) {
    int joint = p->name2id(p->model, mjOBJ_JOINT, names[i]);
    char actuator[256]; snprintf(actuator, sizeof(actuator), "%s_servo", names[i]);
    p->a[i] = p->name2id(p->model, mjOBJ_ACTUATOR, actuator);
    if (joint < 0 || p->a[i] < 0 || p->model->jnt_type[joint] != mjJNT_HINGE) { snprintf(error, error_size, "Missing hinge/servo: %s", names[i]); lamp_close(p); return NULL; }
    p->q[i] = p->model->jnt_qposadr[joint]; p->v[i] = p->model->jnt_dofadr[joint];
    p->lower[i] = lower[i]; p->upper[i] = upper[i];
  }
  p->data = p->make(p->model);
  if (!p->data) { snprintf(error, error_size, "Cannot allocate MuJoCo data"); lamp_close(p); return NULL; }
  p->forward(p->model, p->data);
  return p;
}

static double bounded(LampPhysics* p, int i, double value) { return fmin(p->upper[i], fmax(p->lower[i], value)); }

void lamp_reset(LampPhysics* p, const double* positions) {
  p->reset(p->model, p->data); p->pending = 0; p->pair_count = 0;
  for (int i = 0; i < 5; i++) { p->data->qpos[p->q[i]] = bounded(p, i, positions[i]); p->data->ctrl[p->a[i]] = p->data->qpos[p->q[i]]; }
  p->forward(p->model, p->data);
}

static void observe(LampPhysics* p, LampDiagnostics* diag, int kinematic) {
  if (p->data->ncon > diag->contacts) diag->contacts = p->data->ncon;
  for (int i = 0; i < p->data->ncon; i++) {
    mjContact* contact = &p->data->contact[i];
    if (contact->geom1 >= 0 && contact->geom2 >= 0) {
      int first = p->model->geom_bodyid[contact->geom1], second = p->model->geom_bodyid[contact->geom2];
      if (first > second) { int swap = first; first = second; second = swap; }
      int found = 0;
      for (int j = 0; j < p->pair_count; j++) if (p->pairs[j][0] == first && p->pairs[j][1] == second) found = 1;
      if (!found && p->pair_count < 64) { p->pairs[p->pair_count][0] = first; p->pairs[p->pair_count++][1] = second; }
    }
    diag->penetration_m = fmax(diag->penetration_m, -contact->dist);
    if (!kinematic) { mjtNum f[6]; p->contact_force(p->model, p->data, i, f); diag->contact_force_n = fmax(diag->contact_force_n, sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2])); }
  }
  for (int i = 0; i < 5; i++) diag->max_effort_nm = fmax(diag->max_effort_nm, fabs(p->data->actuator_force[p->a[i]]));
  diag->warnings = 0;
  for (int i = 0; i < mjNWARNING; i++) diag->warnings += p->data->warning[i].number;
}

int lamp_advance(LampPhysics* p, const double* targets, double dt, int kinematic, LampDiagnostics* diag) {
  if (!isfinite(dt) || dt < 0) return 0;
  for (int i = 0; i < 5; i++) if (!isfinite(targets[i])) return 0;
  memset(diag, 0, sizeof(*diag));
  p->pair_count = 0;
  for (int i = 0; i < 5; i++) p->data->ctrl[p->a[i]] = bounded(p, i, targets[i]);
  if (kinematic) {
    for (int i = 0; i < 5; i++) { double next = p->data->ctrl[p->a[i]]; p->data->qvel[p->v[i]] = dt > 0 ? (next-p->data->qpos[p->q[i]])/dt : 0; p->data->qpos[p->q[i]] = next; }
    p->data->time += dt; p->forward(p->model, p->data); observe(p, diag, 1);
  } else {
    p->pending += dt;
    while (p->pending + 1e-12 >= p->model->opt.timestep) {
      p->step(p->model, p->data); observe(p, diag, 0);
      p->pending = fmax(0, p->pending-p->model->opt.timestep);
    }
    p->forward(p->model, p->data);
  }
  for (int i = 0; i < 5; i++) if (!isfinite(p->data->qpos[p->q[i]]) || !isfinite(p->data->qvel[p->v[i]])) return 0;
  return 1;
}

void lamp_positions(LampPhysics* p, double* positions, double* velocity) {
  for (int i = 0; i < 5; i++) { positions[i] = p->data->qpos[p->q[i]]; velocity[i] = p->data->qvel[p->v[i]]; }
}
double lamp_time(LampPhysics* p) { return p->data->time; }
int lamp_body_count(LampPhysics* p) { return p->model->nbody; }
int lamp_pair_count(LampPhysics* p) { return p->pair_count; }
int lamp_pair_body(LampPhysics* p, int pair, int side) { return pair >= 0 && pair < p->pair_count && side >= 0 && side < 2 ? p->pairs[pair][side] : -1; }
const char* lamp_body_name(LampPhysics* p, int id) { return id >= 0 && id < p->model->nbody ? p->id2name(p->model, mjOBJ_BODY, id) : NULL; }
void lamp_body_pose(LampPhysics* p, int id, double* position, double* quaternion) {
  if (id < 0 || id >= p->model->nbody) return;
  memcpy(position, p->data->xpos + 3*id, 3*sizeof(double));
  const mjtNum* q = p->data->xquat + 4*id;
  quaternion[0] = q[1]; quaternion[1] = q[2]; quaternion[2] = q[3]; quaternion[3] = q[0];
}
