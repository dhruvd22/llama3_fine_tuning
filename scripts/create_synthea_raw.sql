-- DDL generated for sample CSV files

CREATE TABLE "immunizations" (
    "DATE" timestamp with time zone,
    "PATIENT" uuid,
    "ENCOUNTER" uuid,
    "CODE" integer,
    "DESCRIPTION" text,
    "BASE_COST" numeric
);

CREATE TABLE "organizations" (
    "Id" uuid,
    "NAME" text,
    "ADDRESS" text,
    "CITY" text,
    "STATE" text,
    "ZIP" text,
    "LAT" numeric,
    "LON" numeric,
    "PHONE" text,
    "REVENUE" numeric,
    "UTILIZATION" integer,
    PRIMARY KEY ("Id")
);

CREATE TABLE "allergies" (
    "START" date,
    "STOP" date,
    "PATIENT" uuid,
    "ENCOUNTER" uuid,
    "CODE" integer,
    "DESCRIPTION" text
);

CREATE TABLE "imaging_studies" (
    "Id" uuid,
    "DATE" timestamp with time zone,
    "PATIENT" uuid,
    "ENCOUNTER" uuid,
    "BODYSITE_CODE" integer,
    "BODYSITE_DESCRIPTION" text,
    "MODALITY_CODE" integer,
    "MODALITY_DESCRIPTION" text,
    "SERIES" integer,
    "INSTANCE" integer,
    PRIMARY KEY ("Id")
);

CREATE TABLE "devices" (
    "DATE" timestamp with time zone,
    "PATIENT" uuid,
    "ENCOUNTER" uuid,
    "CODE" integer,
    "DESCRIPTION" text,
    "UDI" text,
    "STATUS" text
);

CREATE TABLE "careplans" (
    "Id" uuid,
    "START" date,
    "STOP" date,
    "PATIENT" uuid,
    "ENCOUNTER" uuid,
    "CODE" bigint,
    "DESCRIPTION" text,
    "REASONCODE" integer,
    "REASONDESCRIPTION" text,
    PRIMARY KEY ("Id")
);

CREATE TABLE "conditions" (
    "START" date,
    "STOP" date,
    "PATIENT" uuid,
    "ENCOUNTER" uuid,
    "CODE" bigint,
    "DESCRIPTION" text
);

CREATE TABLE "medications" (
    "Id" uuid,
    "START" timestamp with time zone,
    "STOP" timestamp with time zone,
    "PATIENT" uuid,
    "ENCOUNTER" uuid,
    "CODE" integer,
    "DESCRIPTION" text,
    "BASE_COST" numeric,
    "PAYER_COVERAGE" numeric,
    "DISPENSES" integer,
    "TOTALCOST" numeric,
    "REASONCODE" integer,
    "REASONDESCRIPTION" text,
    PRIMARY KEY ("Id")
);

CREATE TABLE "observations" (
    "DATE" timestamp with time zone,
    "PATIENT" uuid,
    "ENCOUNTER" uuid,
    "CODE" text,
    "DESCRIPTION" text,
    "VALUE" text,
    "UNITS" text,
    "TYPE" text
);

CREATE TABLE "encounters" (
    "Id" uuid,
    "START" timestamp with time zone,
    "STOP" timestamp with time zone,
    "PATIENT" uuid,
    "ORGANIZATION" uuid,
    "PROVIDER" uuid,
    "PAYER" uuid,
    "ENCOUNTERCLASS" text,
    "CODE" integer,
    "DESCRIPTION" text,
    "BASE_ENCOUNTER_COST" numeric,
    "TOTAL_CLAIM_COST" numeric,
    "PAYER_COVERAGE" numeric,
    "REASONCODE" integer,
    "REASONDESCRIPTION" text,
    PRIMARY KEY ("Id")
);

-- =========================================
-- Table: payers
-- =========================================
CREATE TABLE payers (
    "Id"                     uuid,
    "NAME"                   text,
    "ADDRESS"                text,
    "CITY"                   text,
    "STATE_HEADQUARTERED"    text,
    "ZIP"                    integer,
    "PHONE"                  text,
    "AMOUNT_COVERED"         numeric,
    "AMOUNT_UNCOVERED"       numeric,
    "REVENUE"                integer,
    "COVERED_ENCOUNTERS"     integer,
    "UNCOVERED_ENCOUNTERS"   integer,
    "COVERED_MEDICATIONS"    integer,
    "UNCOVERED_MEDICATIONS"  integer,
    "COVERED_PROCEDURES"     integer,
    "UNCOVERED_PROCEDURES"   integer,
    "COVERED_IMMUNIZATIONS"  integer,
    "UNCOVERED_IMMUNIZATIONS" integer,
    "UNIQUE_CUSTOMERS"       integer,
    "QOLS_AVG"               numeric,
    "MEMBER_MONTHS"          integer
);
