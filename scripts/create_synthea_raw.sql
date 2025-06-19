/* ======================================================================
   Synthea raw CSV tables 
   ====================================================================== */

/* ---------- allergies ------------------------------------------------ */
CREATE TABLE allergies (
  start             date,
  stop              date,
  patient           varchar,
  encounter         varchar,
  code              varchar,
  system            varchar,
  description       varchar);

/* ---------- careplans ----------------------------------------------- */
CREATE TABLE careplans (
  id                varchar,
  start             date,
  stop              date,
  patient           varchar,
  encounter         varchar,
  code              varchar,
  description       varchar,
  reasoncode        varchar,
  reasondescription varchar
);

/* ---------- claims --------------------------------------------------- */
CREATE TABLE claims (
  id                         varchar,
  patientid                  varchar,
  providerid                 varchar,
  primarypatientinsuranceid  varchar,
  secondarypatientinsuranceid varchar,
  departmentid               integer,
  patientdepartmentid        integer,
  diagnosis1                 varchar,
  diagnosis2                 varchar,
  diagnosis3                 varchar,
  diagnosis4                 varchar,
  diagnosis5                 varchar,
  diagnosis6                 varchar,
  diagnosis7                 varchar,
  diagnosis8                 varchar,
  referringproviderid        varchar,
  appointmentid              varchar,
  currentillnessdate         timestamptz,
  servicedate                timestamptz,
  supervisingproviderid      varchar,
  status1                    varchar,
  status2                    varchar,
  statusp                    varchar,
  outstanding1               float,
  outstanding2               float,
  outstandingp               float,
  lastbilleddate1            timestamptz,
  lastbilleddate2            timestamptz,
  lastbilleddatep            timestamptz,
  healthcareclaimtypeid1     integer,
  healthcareclaimtypeid2     integer
);

/* ---------- claims_transactions ------------------------------------- */
CREATE TABLE claims_transactions (
  id                     varchar,
  claimid                varchar,
  chargeid               integer,
  patientid              varchar,
  type                   varchar,
  amount                 float,
  method                 varchar,
  fromdate               timestamptz,
  todate                 timestamptz,
  placeofservice         varchar,
  procedurecode          varchar,
  modifier1              varchar,
  modifier2              varchar,
  diagnosisref1          integer,
  diagnosisref2          integer,
  diagnosisref3          integer,
  diagnosisref4          integer,
  units                  integer,
  departmentid           integer,
  notes                  varchar,
  unitamount             float,
  transferoutid          integer,
  transfertype           varchar,
  payments               float,
  adjustments            float,
  transfers              float,
  outstanding            float,
  appointmentid          varchar,
  linenote               varchar,
  patientinsuranceid     varchar,
  feescheduleid          integer,
  providerid             varchar,
  supervisingproviderid  varchar
);

/* ---------- conditions ---------------------------------------------- */
CREATE TABLE conditions (
  start             date,
  stop              date,
  patient           varchar,
  encounter         varchar,
  code              varchar,
  description       varchar
);

/* ---------- devices -------------------------------------------------- */
CREATE TABLE devices (
  start             timestamptz,
  stop              timestamptz,
  patient           varchar,
  encounter         varchar,
  code              varchar,
  description       varchar,
  udi               varchar
);

/* ---------- encounters ---------------------------------------------- */
CREATE TABLE encounters (
  id                    varchar,
  start                 timestamptz,
  stop                  timestamptz,
  patient               varchar,
  organization          varchar,
  provider              varchar,
  payer                 varchar,
  encounterclass        varchar,
  code                  varchar,
  description           varchar,
  base_encounter_cost   float,
  total_claim_cost      float,
  payer_coverage        float,
  reasoncode            varchar,
  reasondescription     varchar
);

/* ---------- imaging_studies ----------------------------------------- */
CREATE TABLE imaging_studies (
  id                     varchar,
  date                   timestamptz,
  patient                varchar,
  encounter              varchar,
  series_uid             varchar,
  bodysite_code          varchar,
  bodysite_description   varchar,
  modality_code          varchar,
  modality_description   varchar,
  instance_uid           varchar,
  sop_code               varchar,
  sop_description        varchar,
  procedure_code         varchar
);

/* ---------- immunizations ------------------------------------------- */
CREATE TABLE immunizations (
  date           timestamptz,
  patient        varchar,
  encounter      varchar,
  code           varchar,
  description    varchar,
  base_cost      float
);

/* ---------- medications --------------------------------------------- */
CREATE TABLE medications (
  start             timestamptz,
  stop              timestamptz,
  patient           varchar,
  payer             varchar,
  encounter         varchar,
  code              varchar,
  description       varchar,
  base_cost         float,
  payer_coverage    float,
  dispenses         integer,
  totalcost         float,
  reasoncode        varchar,
  reasondescription varchar
);

/* ---------- observations -------------------------------------------- */
CREATE TABLE observations (
  date           timestamptz,
  patient        varchar,
  encounter      varchar,
  category       varchar,
  code           varchar,
  description    varchar,
  value          varchar,
  units          varchar,
  type           varchar
);

/* ---------- organizations ------------------------------------------- */
CREATE TABLE organizations (
  id          varchar,
  name        varchar,
  address     varchar,
  city        varchar,
  state       varchar,
  zip         varchar,
  lat         float,
  lon         float,
  phone       varchar,
  revenue     float,
  utilization integer
);

/* ---------- patients ------------------------------------------------- */
CREATE TABLE patients (
  id                  varchar,
  birthdate           date,
  deathdate           date,
  ssn                 varchar,
  drivers             varchar,
  passport            varchar,
  prefix              varchar,
  first               varchar,
  last                varchar,
  suffix              varchar,
  maiden              varchar,
  marital             varchar,
  race                varchar,
  ethnicity           varchar,
  gender              varchar,
  birthplace          varchar,
  address             varchar,
  city                varchar,
  state               varchar,
  county              varchar,
  zip                 varchar,
  lat                 float,
  lon                 float,
  healthcare_expenses float,
  healthcare_coverage float
);

/* ---------- payer_transitions --------------------------------------- */
CREATE TABLE payer_transitions (
  patient        varchar,
  memberid       varchar,
  start_year     timestamptz,
  end_year       timestamptz,
  payer          varchar,
  secondary_payer varchar,
  ownership      varchar,
  ownername      varchar
);

/* ---------- payers --------------------------------------------------- */
CREATE TABLE payers (
  id                       varchar,
  name                     varchar,
  address                  varchar,
  city                     varchar,
  state_headquartered      varchar,
  zip                      integer,
  phone                    varchar,
  amount_covered           float,
  amount_uncovered         float,
  revenue                  float,
  covered_encounters       integer,
  uncovered_encounters     integer,
  covered_medications      integer,
  uncovered_medications    integer,
  covered_procedures       integer,
  uncovered_procedures     integer,
  covered_immunizations    integer,
  uncovered_immunizations  integer,
  unique_customers         integer,
  qols_avg                 float,
  member_months            integer
);

/* ---------- procedures ---------------------------------------------- */
CREATE TABLE procedures (
  start             timestamptz,
  stop              timestamptz,
  patient           varchar,
  encounter         varchar,
  code              varchar,
  description       varchar,
  base_cost         float,
  reasoncode        varchar,
  reasondescription varchar
);

/* ---------- providers ------------------------------------------------ */
CREATE TABLE providers (
  id           varchar,
  organization varchar,
  name         varchar,
  gender       varchar,
  speciality   varchar,
  address      varchar,
  city         varchar,
  state        varchar,
  zip          varchar,
  lat          float,
  lon          float,
  utilization  integer
);

/* ---------- supplies ------------------------------------------------- */
CREATE TABLE supplies (
  date        date,
  patient     varchar,
  encounter   varchar,
  code        varchar,
  description varchar,
  quantity    integer
);
