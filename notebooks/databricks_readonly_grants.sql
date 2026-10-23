-- Unity Catalog read-only access for agentic-gis-databricks.
-- Run as a catalog/schema owner or metastore admin. Replace names with your own.
-- The app should connect as a principal that has ONLY these privileges (no MODIFY/CREATE/ALL PRIVILEGES).

-- 1. A group (or service principal) for the app's identity
--    (create the group in the account console; add the user or service principal to it)

-- 2. Catalog and schema usage
GRANT USE CATALOG ON CATALOG dsny_sandbox TO `agentic-gis-readers`;
GRANT USE SCHEMA ON SCHEMA dsny_sandbox.egpmo_sandbox TO `agentic-gis-readers`;

-- 3. SELECT on exactly the tables registered in config/datasets(.local).yaml
GRANT SELECT ON TABLE dsny_sandbox.egpmo_sandbox.catch_basins_exploded_final TO `agentic-gis-readers`;
GRANT SELECT ON TABLE dsny_sandbox.egpmo_sandbox.catch_basin_records_final TO `agentic-gis-readers`;

-- 4. Verify
SHOW GRANTS `agentic-gis-readers` ON SCHEMA dsny_sandbox.egpmo_sandbox;

-- 5. SQL warehouse: grant "Can use" in the warehouse's Permissions dialog (not SQL).

-- Optional: sensitive columns can be protected in Unity Catalog with column masks / row filters;
-- also mark them `sensitive: true` in the registry so the agent never selects or describes them.
