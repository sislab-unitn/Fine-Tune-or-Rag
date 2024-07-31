import datetime
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set


# taxi "table" only has a set of colours, a set of brands ("types"), and a set of possible phone numbers
UNAVAIL_SLOTS = [
    "hotel-bookday",
    "hotel-bookpeople", # hotel-bookpeople based on presence of "single/double/family"?
    "hotel-bookstay",
    "restaurant-bookday",
    "restaurant-bookpeople",
    "restaurant-booktime",
    "taxi-arriveby",
    "taxi-departure",
    "taxi-destination",
    "taxi-leaveat",
    "train-bookpeople"
]

# "bus/train-leaveat" ("I'd like to leave AFTER ..." vs "I'd like to leave BEFORE ...")
EQ_SLOTS = [
    "attraction-area",
    "attraction-name",
    "attraction-type",
    "bus-day",
    "bus-departure",
    "bus-destination",
    "hotel-area",
    "hotel-internet",
    "hotel-name",
    "hotel-parking",
    "hotel-pricerange",
    "hotel-stars", # similar to "-leaveat" (could be in LEQ or GEQ)
    "hotel-type",
    "restaurant-area",
    "restaurant-food",
    "restaurant-name",
    "restaurant-pricerange",
    "train-day",
    "train-departure",
    "train-destination",
]
GEQ_SLOTS = [
    "bus-leaveat",  # could be in LEQ
    "train-leaveat" # could be in LEQ
]
LEQ_SLOTS = [
    "bus-arriveby",
    "train-arriveby"
]

AVAIL_SLOTS = EQ_SLOTS + GEQ_SLOTS + LEQ_SLOTS

SLOT_MAP = {
    "arriveby": "arriveBy",
    "leaveat": "leaveAt",
}

# TIME_SLOTS = ["bus-leaveat", "bus-arriveby", "restaurant-booktime", "taxi-arriveby", "taxi-leaveat", "train-arriveby", "train-leaveat"]
# INT_SLOTS = ["hotel-bookpeople", "hotel-bookstay", "hotel-stars", "restaurant-bookpeople", "train-bookpeople"]
TIME_SLOTS = ["bus-leaveat", "bus-arriveby", "train-arriveby", "train-leaveat"]
INT_SLOTS = ["hotel-stars"]


TIME_PATTERNS = [
    (r"([0-2]?[0-9])(?:\:|\.)?([0-9][0-9])", r"\1:\2"),
    (r"([0-2]?[0-9])", r"\1:00")
]

EMPTY_FIELD_VALUES = ["", "?"]

LONG_FIELDS = ["openhours", "introduction", "signature"]

def str_to_time(time_str: str):
    orig_time_str = time_str
    try:
        # TODO check regex for multiwoz 2.1
        for patt, sub in TIME_PATTERNS:
            comp_patt = re.compile(patt)
            match = comp_patt.match(time_str)
            if match is not None:
                time_str = re.sub(patt, sub, match[0])
                break
        hour, minute = time_str.split(":")
        time = datetime.time(hour=int(hour)%24, minute=int(minute))
    except:
        time = None
    return time

SEP1 = "<SEP1>"
SEP2 = "<SEP2>"

def get_top_k_values(table: str, entries: List[Dict[str,Any]], k: int) -> Dict[str,List[Any]]:
    value_counts = {}
    for entry in entries:
        for field, value in entry.items():
            # skip "null" values ("", "?", ..) for top k
            if value not in EMPTY_FIELD_VALUES:
                # need to be hashable
                if field == "location":
                    # [long, lat]
                    value = SEP1.join([str(v) for v in value])
                if table == "hotel" and field == "price":
                    # {single: 50, double: 90, ...}
                    value = SEP1.join([f"{k}{SEP2}{v}" for k, v in value.items()]) #type: ignore

                if field not in value_counts:
                    value_counts[field] = {}
                if value not in value_counts[field]:
                    value_counts[field][value] = 0
                value_counts[field][value] += 1
    top_k_values = {}
    for field, values in value_counts.items():
        vc_list = [(value, count) for value, count in values.items()]
        sorted_vc_list = sorted(vc_list, key=lambda x: x[1], reverse=True)
        top_k, _ = zip(*sorted_vc_list[:min(k, len(sorted_vc_list))])
        top_k_values[field] = list(top_k)
    # restore location and hotel price
    for field, value in top_k_values.items():
        if field == "location":
            top_k_values[field] = [[e for e in v.split(SEP1)] for v in value]
        if table == "hotel" and field == "price":
            top_k_values[field] = [{e.split(SEP2)[0]: e.split(SEP2)[1] for e in v.split(SEP1)} for v in value] #type: ignore

    return top_k_values

def summarise_query_result(result: Dict[str, List[Dict[str,Any]]], k: int) -> Dict[str, List[Dict[str,Any]]]:
    """
    From list of results to list of top k most common values for each field.
    """
    summarised = {}
    for table, entries in result.items():
        if len(entries) > 1:
            top_k_values = get_top_k_values(table, entries, k)
            # keep same format
            summarised[table] = [top_k_values]
        else:
            summarised[table] = entries
    return summarised


class MwozDataBase:
    def __init__(self, db_path: str, accept_dontcare: bool=False):
        self.db: Dict[str, List[Dict[str,Any]]] = {}
        self.accept_dontcare: bool = accept_dontcare
        self._load_db(db_path)

    def _load_db(self, db_path: str):
        db_files_paths = [os.path.join(db_path, name) for name in os.listdir(db_path) if "_db.json" in name]
        for db_file_path in db_files_paths:
            domain = Path(db_file_path).name.split("_")[0]
            if domain == "taxi":
                # always skip, no db for taxi
                continue
            # TODO analysis for hospital and police
            # skip hospital and police domains
            if "hospital" in domain or "police" in domain:
                continue
            # load data for domain
            with open(db_file_path, "r") as f:
                self.db[domain] = json.load(f)

    def query(self, query: Dict[str,str], exclude_long_fields: bool = False) -> Dict[str, List[Dict[str,Any]]]:
        query_domains = set([slot.split("-")[0] for slot in query.keys()])
        # always remove taxi
        if "taxi" in query_domains:
            query_domains.remove("taxi")
        result = {domain: [] for domain in query_domains}
        for domain in query_domains:
            # consider elements of domain for query
            dom_query = {slot: value for slot, value in query.items() if slot.split("-")[0] == domain}
            dom_query = self._filter_query(dom_query)
            # TODO check what happens: issue with bus
            if domain not in self.db:
                continue
            for entry in self.db[domain]:
                if self._is_match(entry, dom_query):
                    if exclude_long_fields:
                        entry = self._remove_long_fields(entry)
                    result[domain].append(entry)
        return result

    def _remove_long_fields(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        new_entry = {}
        for field, value in entry.items():
            if field not in LONG_FIELDS:
                new_entry[field] = value
        return new_entry

    def _filter_query(self, query: Dict[str, str]) -> Dict[str, str]:
        # depending on the domain, some slots cannot be checked (no information in DB)
        return {slot:value for slot, value in query.items() if slot not in UNAVAIL_SLOTS}


    def _is_match(self, entry: Dict[str, Any], query: Dict[str, str]) -> bool:
        match = True
        for domain_slot, value in query.items():
            assert " " not in domain_slot
            slot = domain_slot.split("-")[1]
            # some slot names are not identical in the database (e.g. arriveby -> arriveBy)
            if slot in SLOT_MAP:
                slot = SLOT_MAP[slot]
            if not self._is_acceptable(domain_slot, value, entry[slot]):
                match = False
                break
        return match

    def _is_acceptable(self, slot: str , value, entry_value) -> bool:
        if value == "dontcare":
            return self.accept_dontcare

        # cast to correct type
        if slot in INT_SLOTS:
            if "|" in value:
                # if alternatives with "|" take first
                value = value.split("|")[0]
            value = int(value)
            entry_value = int(entry_value)
        elif slot in TIME_SLOTS:
            orig_value = value
            value = str_to_time(value)
            entry_value = str_to_time(entry_value)
            # TODO manage some cases
            if value is None:
                return False

        if slot in EQ_SLOTS:
            acceptable = entry_value == value
        elif slot in LEQ_SLOTS:
            acceptable = entry_value <= value # type: ignore
        else: # slot in GEQ_SLOTS
            acceptable = entry_value >= value # type: ignore

        return acceptable

    def get_all_fields(self) -> Dict[str, Set[str]]:
        all_fields = {}
        for table, entries in self.db.items():
            all_fields[table] = set()
            for entry in entries:
                for field in entry.keys():
                    all_fields[table].add(field)
        return all_fields
